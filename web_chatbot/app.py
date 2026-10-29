"""
LangChain + GCP Vertex AI (ADC 인증) & 로컬 Ollama(gemma4:e2b) 하이브리드 챗봇 (FastAPI 웹 서비스)
실시간 구글 검색(Google Search Grounding) ON/OFF 토글 & 토큰/비용 카운터 대시보드 내장

실행 방법:
    python app.py
    또는
    uv run python app.py
    (브라우저에서 http://127.0.0.1:8000 접속)
"""

import os
import re
import sys
import time
import uuid
import subprocess
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
import google.auth
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import PostgresStore

# .env 파일의 DATABASE_URL, GOOGLE_CSE_API_KEY, GOOGLE_CSE_ID 등을 로드
load_dotenv()

# Windows 콘솔 한글/특수문자 인코딩 보정
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ==============================================================================
# [0] 저장소(PostgreSQL) 및 LangGraph Store 캐시 설정
# ==============================================================================
# 대화 내용(LangGraph 체크포인트)과 세션 토큰/비용 통계를 PostgreSQL에 저장합니다.
#   - 대화 메시지: PostgresSaver가 만드는 checkpoints 계열 테이블 (thread_id 기준)
#   - 토큰/비용 누적: chat_sessions 테이블 (이 파일에서 직접 생성)
#   - 검색 결과 캐시: LangGraph Store (PostgresStore / InMemoryStore)
# DATABASE_URL 예: postgresql://postgres:postgres@localhost:5432/chatbot
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

CHECKPOINTER = None  # PostgresSaver (DATABASE_URL 미설정/연결 실패 시 InMemorySaver로 대체)
DB_POOL: ConnectionPool | None = None
STORAGE_BACKEND = "memory"
SESSION_STATS: dict[str, dict] = {}  # 메모리 대체 모드에서만 사용
MEMORY_USAGE_LOGS: list[dict] = []    # 메모리 모드에서의 턴별 모델 사용량 기록

# 모델별 가격 정책 (100만 토큰당 USD 단가 및 환율)
USD_TO_KRW = 1380.0
MODEL_PRICING = {
    "gemini-2.5-flash": {
        "label": "Gemini 2.5 Flash (Medium)",
        "input_per_1m": 0.075,
        "output_per_1m": 0.30,
    },
    "gemini-2.5-flash-low": {
        "label": "Gemini 2.5 빠른 추론 (Low)",
        "input_per_1m": 0.075,
        "output_per_1m": 0.30,
    },
    "gemini-2.5-thinking": {
        "label": "Gemini 2.5 사고 모델 (High)",
        "input_per_1m": 0.15,
        "output_per_1m": 0.60,
    },
    "gemini-2.5-pro": {
        "label": "Gemini 2.5 Pro",
        "input_per_1m": 1.25,
        "output_per_1m": 5.00,
    },
    "ollama:gemma4:e2b": {
        "label": "로컬 Ollama (gemma4:e2b)",
        "input_per_1m": 0.0,
        "output_per_1m": 0.0,
    },
}

# LangGraph Store 기반 구글 검색 결과 캐시 설정
GLOBAL_STORE: BaseStore = InMemoryStore()
SEARCH_CACHE_NAMESPACE = ("cache", "google_search")
SEARCH_CACHE_TTL = 1800  # 캐시 유효 시간: 30분 (1,800초)
CACHE_STATS = {"hits": 0, "misses": 0, "total_calls": 0}


def _empty_stats() -> dict:
    return {"total_input": 0, "total_output": 0, "total_cost_krw": 0.0, "total_cost_usd": 0.0}


def init_storage():
    """PostgreSQL 연결 풀, PostgresSaver, LangGraph Store를 초기화하고 필요한 테이블을 생성합니다."""
    global CHECKPOINTER, DB_POOL, STORAGE_BACKEND, GLOBAL_STORE
    if not DATABASE_URL:
        print("⚠️  DATABASE_URL이 없어 InMemorySaver 및 InMemoryStore로 실행합니다. (재시작 시 대화가 사라집니다)")
        CHECKPOINTER = InMemorySaver()
        GLOBAL_STORE = InMemoryStore()
        return
    try:
        # PostgresSaver는 autocommit=True, row_factory=dict_row 연결이 필요합니다.
        # check=ConnectionPool.check_connection 및 max_idle 설정으로 유휴 SSL 연결 끊김 자동 복구
        DB_POOL = ConnectionPool(
            conninfo=DATABASE_URL,
            min_size=1,
            max_size=10,
            max_idle=30.0,
            check=ConnectionPool.check_connection,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
            open=False,
        )
        DB_POOL.open(wait=True, timeout=10)
        CHECKPOINTER = PostgresSaver(DB_POOL)
        CHECKPOINTER.setup()  # checkpoints 테이블 생성/마이그레이션

        # LangGraph PostgresStore 초기화 (검색 캐시를 PostgreSQL에 저장)
        try:
            GLOBAL_STORE = PostgresStore(DB_POOL)
            GLOBAL_STORE.setup()
            print("✅ LangGraph PostgresStore 연결 완료: 검색 캐시를 DB에 영구 저장합니다.")
        except Exception as e_store:
            print(f"⚠️  PostgresStore 초기화 실패({e_store}). InMemoryStore로 대체합니다.")
            GLOBAL_STORE = InMemoryStore()

        with DB_POOL.connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    thread_id      TEXT PRIMARY KEY,
                    title          TEXT,
                    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
                    total_input    BIGINT NOT NULL DEFAULT 0,
                    total_output   BIGINT NOT NULL DEFAULT 0,
                    total_cost_krw DOUBLE PRECISION NOT NULL DEFAULT 0,
                    total_cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute("ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS title TEXT")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS token_usage_logs (
                    id             BIGSERIAL PRIMARY KEY,
                    thread_id      TEXT NOT NULL,
                    model_key      TEXT NOT NULL,
                    model_label    TEXT NOT NULL,
                    input_tokens   BIGINT NOT NULL DEFAULT 0,
                    output_tokens  BIGINT NOT NULL DEFAULT 0,
                    cost_usd       DOUBLE PRECISION NOT NULL DEFAULT 0,
                    cost_krw       DOUBLE PRECISION NOT NULL DEFAULT 0,
                    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_token_usage_created ON token_usage_logs (created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_token_usage_model ON token_usage_logs (model_key)")
        STORAGE_BACKEND = "postgres"
        print("✅ PostgreSQL 연결 완료: 대화 내용을 DB에 저장합니다.")
    except Exception as e:
        print(f"⚠️  PostgreSQL 연결 실패({e}). InMemorySaver 및 InMemoryStore로 대체합니다.")
        if DB_POOL is not None:
            DB_POOL.close()
            DB_POOL = None
        CHECKPOINTER = InMemorySaver()
        GLOBAL_STORE = InMemoryStore()
        STORAGE_BACKEND = "memory"


def create_session(thread_id: str, title: str | None = None):
    """새 세션 행을 생성합니다."""
    if STORAGE_BACKEND == "postgres":
        with DB_POOL.connection() as conn:
            conn.execute(
                "INSERT INTO chat_sessions (thread_id, title) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (thread_id, title),
            )
    else:
        st = _empty_stats()
        if title:
            st["title"] = title
        SESSION_STATS.setdefault(thread_id, st)


def update_session_title_if_empty(thread_id: str, title: str):
    """세션 제목이 없을 경우 사용자 첫 질문 등으로 제목을 설정합니다."""
    if not title:
        return
    clean_title = title.strip().replace("\n", " ")
    if len(clean_title) > 35:
        clean_title = clean_title[:35] + "..."
    if STORAGE_BACKEND == "postgres":
        with DB_POOL.connection() as conn:
            conn.execute(
                """
                UPDATE chat_sessions 
                SET title = %s 
                WHERE thread_id = %s AND (title IS NULL OR title = '' OR title = '새로운 대화');
                """,
                (clean_title, thread_id),
            )
    else:
        if thread_id in SESSION_STATS and (not SESSION_STATS[thread_id].get("title") or SESSION_STATS[thread_id].get("title") == "새로운 대화"):
            SESSION_STATS[thread_id]["title"] = clean_title


def update_session_title(thread_id: str, new_title: str):
    """세션 제목을 강제로 새 이름으로 변경합니다."""
    clean_title = new_title.strip().replace("\n", " ")[:60]
    if not clean_title:
        return
    if STORAGE_BACKEND == "postgres":
        with DB_POOL.connection() as conn:
            conn.execute(
                """
                UPDATE chat_sessions 
                SET title = %s, updated_at = now()
                WHERE thread_id = %s;
                """,
                (clean_title, thread_id),
            )
    else:
        if thread_id in SESSION_STATS:
            SESSION_STATS[thread_id]["title"] = clean_title


def list_sessions() -> list[dict]:
    """저장된 전체 세션 목록을 최신 순으로 반환합니다."""
    if STORAGE_BACKEND == "postgres":
        with DB_POOL.connection() as conn:
            # checkpoints에는 있지만 chat_sessions에 없는 세션이 있다면 자동 동기화
            conn.execute(
                """
                INSERT INTO chat_sessions (thread_id, total_input, total_output, total_cost_krw, total_cost_usd, updated_at)
                SELECT DISTINCT thread_id, 0, 0, 0, 0, now()
                FROM checkpoints
                WHERE thread_id NOT IN (SELECT thread_id FROM chat_sessions)
                ON CONFLICT (thread_id) DO NOTHING;
                """
            )
            rows = conn.execute(
                """
                SELECT thread_id, title, total_input, total_output, total_cost_krw, total_cost_usd, updated_at
                FROM chat_sessions
                ORDER BY updated_at DESC
                LIMIT 50;
                """
            ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            if not d.get("title") or d.get("title") == "새로운 대화":
                first_q = get_first_question(d["thread_id"])
                if first_q:
                    d["title"] = first_q
                    update_session_title_if_empty(d["thread_id"], first_q)
                else:
                    d["title"] = d.get("title") or f"대화 ({d['thread_id']})"
            results.append(d)
        return results
    else:
        results = []
        for tid, s in sorted(SESSION_STATS.items(), key=lambda x: x[0], reverse=True):
            results.append({
                "thread_id": tid,
                "title": s.get("title") or f"대화 ({tid})",
                "total_input": s.get("total_input", 0),
                "total_output": s.get("total_output", 0),
                "total_cost_krw": s.get("total_cost_krw", 0.0),
                "total_cost_usd": s.get("total_cost_usd", 0.0),
                "updated_at": "",
            })
        return results


def get_first_question(thread_id: str) -> str:
    """체크포인트에서 첫 번째 사용자 질문을 찾아 제목으로 반환합니다."""
    msgs = _load_messages(thread_id)
    for m in msgs:
        if m.get("type") == "human" and m.get("content"):
            q = m["content"].strip().replace("\n", " ")
            return q[:35] + ("..." if len(q) > 35 else "")
    return ""


def delete_session(thread_id: str):
    """세션과 관련 체크포인트를 DB/메모리에서 삭제합니다."""
    if STORAGE_BACKEND == "postgres":
        with DB_POOL.connection() as conn:
            conn.execute("DELETE FROM chat_sessions WHERE thread_id = %s;", (thread_id,))
            conn.execute("DELETE FROM checkpoints WHERE thread_id = %s;", (thread_id,))
            conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s;", (thread_id,))
            conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s;", (thread_id,))
    else:
        SESSION_STATS.pop(thread_id, None)


def get_session_stats(thread_id: str) -> dict:
    """세션의 누적 토큰/비용 통계를 반환합니다."""
    if STORAGE_BACKEND == "postgres":
        with DB_POOL.connection() as conn:
            row = conn.execute(
                "SELECT total_input, total_output, total_cost_krw, total_cost_usd "
                "FROM chat_sessions WHERE thread_id = %s",
                (thread_id,),
            ).fetchone()
        return dict(row) if row else _empty_stats()
    return SESSION_STATS.get(thread_id, _empty_stats())


def resolve_model_key_and_label(model_choice: str, reasoning_effort: str = "medium") -> tuple[str, str]:
    """선택된 모델과 추론 수준에 따른 고유 모델 키와 표시 이름을 반환합니다."""
    m = (model_choice or "gemini").lower()
    effort = (reasoning_effort or "medium").lower()
    if "ollama" in m:
        return "ollama:gemma4:e2b", MODEL_PRICING["ollama:gemma4:e2b"]["label"]
    if effort == "high":
        return "gemini-2.5-thinking", MODEL_PRICING["gemini-2.5-thinking"]["label"]
    if effort == "low":
        return "gemini-2.5-flash-low", MODEL_PRICING["gemini-2.5-flash-low"]["label"]
    return "gemini-2.5-flash", MODEL_PRICING["gemini-2.5-flash"]["label"]


def add_session_stats(thread_id: str, turn: dict) -> dict:
    """이번 턴의 토큰/비용을 세션 누적값 및 모델별 사용 로그(token_usage_logs)에 기록합니다."""
    model_key = turn.get("model_key", "gemini-2.5-flash")
    model_label = turn.get("model_label", "Gemini 2.5 Flash (Medium)")

    if STORAGE_BACKEND == "postgres":
        with DB_POOL.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO chat_sessions (thread_id, total_input, total_output, total_cost_krw, total_cost_usd)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (thread_id) DO UPDATE SET
                    total_input    = chat_sessions.total_input    + EXCLUDED.total_input,
                    total_output   = chat_sessions.total_output   + EXCLUDED.total_output,
                    total_cost_krw = chat_sessions.total_cost_krw + EXCLUDED.total_cost_krw,
                    total_cost_usd = chat_sessions.total_cost_usd + EXCLUDED.total_cost_usd,
                    updated_at     = now()
                RETURNING total_input, total_output, total_cost_krw, total_cost_usd
                """,
                (thread_id, turn["input_tokens"], turn["output_tokens"], turn["cost_krw"], turn["cost_usd"]),
            ).fetchone()
            conn.execute(
                """
                INSERT INTO token_usage_logs (thread_id, model_key, model_label, input_tokens, output_tokens, cost_usd, cost_krw)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    thread_id,
                    model_key,
                    model_label,
                    turn["input_tokens"],
                    turn["output_tokens"],
                    turn["cost_usd"],
                    turn["cost_krw"],
                ),
            )
        return dict(row)

    sess = SESSION_STATS.setdefault(thread_id, _empty_stats())
    sess["total_input"] += turn["input_tokens"]
    sess["total_output"] += turn["output_tokens"]
    sess["total_cost_krw"] += turn["cost_krw"]
    sess["total_cost_usd"] += turn["cost_usd"]
    MEMORY_USAGE_LOGS.append({
        "thread_id": thread_id,
        "model_key": model_key,
        "model_label": model_label,
        "input_tokens": turn["input_tokens"],
        "output_tokens": turn["output_tokens"],
        "cost_usd": turn["cost_usd"],
        "cost_krw": turn["cost_krw"],
        "created_at": datetime.now(),
    })
    return sess


def get_token_analytics(period: str = "daily") -> dict:
    """
    모델별 누적 토큰 사용량 및 일별(daily) / 주간별(weekly) / 월별(monthly) 사용량 통계를 반환합니다.
    """
    period = period if period in ("daily", "weekly", "monthly", "all") else "daily"
    by_model = []
    timeline = []

    if STORAGE_BACKEND == "postgres":
        with DB_POOL.connection() as conn:
            # 1) token_usage_logs가 비어있고 기존 chat_sessions에 누적 데이터가 있다면 초기 마이그레이션 보정
            cnt_row = conn.execute("SELECT COUNT(*) AS cnt FROM token_usage_logs").fetchone()
            if cnt_row and cnt_row["cnt"] == 0:
                conn.execute(
                    """
                    INSERT INTO token_usage_logs (thread_id, model_key, model_label, input_tokens, output_tokens, cost_usd, cost_krw, created_at)
                    SELECT thread_id, 'gemini-2.5-flash', 'Gemini 2.5 Flash (Medium)', total_input, total_output, total_cost_usd, total_cost_krw, updated_at
                    FROM chat_sessions
                    WHERE (total_input + total_output) > 0
                    """
                )

            # 2) 기간 필터 조건 설정
            where_clause = ""
            if period == "daily":
                where_clause = "WHERE created_at >= CURRENT_DATE"
            elif period == "weekly":
                where_clause = "WHERE created_at >= (now() - INTERVAL '7 days')"
            elif period == "monthly":
                where_clause = "WHERE created_at >= (now() - INTERVAL '30 days')"

            # 3) 선택된 기간의 모델별 누적 통계
            m_rows = conn.execute(
                f"""
                SELECT
                    model_key,
                    MAX(model_label) AS model_label,
                    COUNT(*) AS call_count,
                    COALESCE(SUM(input_tokens), 0) AS input_tokens,
                    COALESCE(SUM(output_tokens), 0) AS output_tokens,
                    COALESCE(SUM(input_tokens + output_tokens), 0) AS total_tokens,
                    COALESCE(SUM(cost_usd), 0) AS cost_usd,
                    COALESCE(SUM(cost_krw), 0) AS cost_krw
                FROM token_usage_logs
                {where_clause}
                GROUP BY model_key
                ORDER BY total_tokens DESC
                """
            ).fetchall()
            for r in m_rows:
                by_model.append({
                    "model_key": r["model_key"],
                    "model_label": r["model_label"],
                    "call_count": int(r["call_count"]),
                    "input_tokens": int(r["input_tokens"]),
                    "output_tokens": int(r["output_tokens"]),
                    "total_tokens": int(r["total_tokens"]),
                    "cost_usd": round(float(r["cost_usd"]), 6),
                    "cost_krw": round(float(r["cost_krw"]), 4),
                })

            # 4) 기간별 추이 (일별: 최근 7일, 주간별: 최근 8주, 월별: 최근 6개월)
            if period == "daily" or period == "all":
                t_rows = conn.execute(
                    """
                    SELECT
                        to_char(created_at AT TIME ZONE 'Asia/Seoul', 'MM-DD') AS bucket,
                        COUNT(*) AS call_count,
                        COALESCE(SUM(input_tokens), 0) AS input_tokens,
                        COALESCE(SUM(output_tokens), 0) AS output_tokens,
                        COALESCE(SUM(input_tokens + output_tokens), 0) AS total_tokens,
                        COALESCE(SUM(cost_krw), 0) AS cost_krw,
                        COALESCE(SUM(cost_usd), 0) AS cost_usd
                    FROM token_usage_logs
                    WHERE created_at >= (now() - INTERVAL '14 days')
                    GROUP BY bucket
                    ORDER BY bucket DESC
                    LIMIT 10
                    """
                ).fetchall()
            elif period == "weekly":
                t_rows = conn.execute(
                    """
                    SELECT
                        to_char(date_trunc('week', created_at AT TIME ZONE 'Asia/Seoul'), 'YYYY-MM-DD') || ' 주간' AS bucket,
                        COUNT(*) AS call_count,
                        COALESCE(SUM(input_tokens), 0) AS input_tokens,
                        COALESCE(SUM(output_tokens), 0) AS output_tokens,
                        COALESCE(SUM(input_tokens + output_tokens), 0) AS total_tokens,
                        COALESCE(SUM(cost_krw), 0) AS cost_krw,
                        COALESCE(SUM(cost_usd), 0) AS cost_usd
                    FROM token_usage_logs
                    WHERE created_at >= (now() - INTERVAL '8 weeks')
                    GROUP BY bucket
                    ORDER BY bucket DESC
                    LIMIT 8
                    """
                ).fetchall()
            else:  # monthly
                t_rows = conn.execute(
                    """
                    SELECT
                        to_char(created_at AT TIME ZONE 'Asia/Seoul', 'YYYY-MM') AS bucket,
                        COUNT(*) AS call_count,
                        COALESCE(SUM(input_tokens), 0) AS input_tokens,
                        COALESCE(SUM(output_tokens), 0) AS output_tokens,
                        COALESCE(SUM(input_tokens + output_tokens), 0) AS total_tokens,
                        COALESCE(SUM(cost_krw), 0) AS cost_krw,
                        COALESCE(SUM(cost_usd), 0) AS cost_usd
                    FROM token_usage_logs
                    WHERE created_at >= (now() - INTERVAL '12 months')
                    GROUP BY bucket
                    ORDER BY bucket DESC
                    LIMIT 12
                    """
                ).fetchall()

            for r in t_rows:
                timeline.append({
                    "bucket": r["bucket"],
                    "call_count": int(r["call_count"]),
                    "input_tokens": int(r["input_tokens"]),
                    "output_tokens": int(r["output_tokens"]),
                    "total_tokens": int(r["total_tokens"]),
                    "cost_krw": round(float(r["cost_krw"]), 4),
                    "cost_usd": round(float(r["cost_usd"]), 6),
                })

            # 5) 요약 카드용(오늘 / 최근 7일 / 최근 30일 / 전체 누적)
            summary_row = conn.execute(
                """
                SELECT
                    COALESCE(SUM(CASE WHEN created_at >= CURRENT_DATE THEN input_tokens + output_tokens ELSE 0 END), 0) AS daily_tokens,
                    COALESCE(SUM(CASE WHEN created_at >= CURRENT_DATE THEN cost_krw ELSE 0 END), 0) AS daily_cost_krw,
                    COALESCE(SUM(CASE WHEN created_at >= (now() - INTERVAL '7 days') THEN input_tokens + output_tokens ELSE 0 END), 0) AS weekly_tokens,
                    COALESCE(SUM(CASE WHEN created_at >= (now() - INTERVAL '7 days') THEN cost_krw ELSE 0 END), 0) AS weekly_cost_krw,
                    COALESCE(SUM(CASE WHEN created_at >= (now() - INTERVAL '30 days') THEN input_tokens + output_tokens ELSE 0 END), 0) AS monthly_tokens,
                    COALESCE(SUM(CASE WHEN created_at >= (now() - INTERVAL '30 days') THEN cost_krw ELSE 0 END), 0) AS monthly_cost_krw,
                    COALESCE(SUM(input_tokens + output_tokens), 0) AS all_tokens,
                    COALESCE(SUM(cost_krw), 0) AS all_cost_krw,
                    COALESCE(SUM(cost_usd), 0) AS all_cost_usd
                FROM token_usage_logs
                """
            ).fetchone()
            period_summary = {
                "daily": {"tokens": int(summary_row["daily_tokens"]), "cost_krw": round(float(summary_row["daily_cost_krw"]), 4)},
                "weekly": {"tokens": int(summary_row["weekly_tokens"]), "cost_krw": round(float(summary_row["weekly_cost_krw"]), 4)},
                "monthly": {"tokens": int(summary_row["monthly_tokens"]), "cost_krw": round(float(summary_row["monthly_cost_krw"]), 4)},
                "all": {
                    "tokens": int(summary_row["all_tokens"]),
                    "cost_krw": round(float(summary_row["all_cost_krw"]), 4),
                    "cost_usd": round(float(summary_row["all_cost_usd"]), 6),
                },
            }
    else:
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = now - timedelta(days=7)
        month_start = now - timedelta(days=30)

        def _in_period(dt: datetime) -> bool:
            if period == "daily":
                return dt >= today_start
            if period == "weekly":
                return dt >= week_start
            if period == "monthly":
                return dt >= month_start
            return True

        filtered = [lg for lg in MEMORY_USAGE_LOGS if _in_period(lg["created_at"])]
        agg_model: dict[str, dict] = {}
        for lg in filtered:
            mk = lg["model_key"]
            entry = agg_model.setdefault(mk, {
                "model_key": mk,
                "model_label": lg["model_label"],
                "call_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "cost_krw": 0.0,
            })
            entry["call_count"] += 1
            entry["input_tokens"] += lg["input_tokens"]
            entry["output_tokens"] += lg["output_tokens"]
            entry["total_tokens"] += lg["input_tokens"] + lg["output_tokens"]
            entry["cost_usd"] += lg["cost_usd"]
            entry["cost_krw"] += lg["cost_krw"]
        by_model = sorted(agg_model.values(), key=lambda x: x["total_tokens"], reverse=True)

        agg_time: dict[str, dict] = {}
        for lg in MEMORY_USAGE_LOGS:
            dt = lg["created_at"]
            if period == "weekly":
                monday = dt - timedelta(days=dt.weekday())
                b = monday.strftime("%Y-%m-%d") + " 주간"
            elif period == "monthly":
                b = dt.strftime("%Y-%m")
            else:
                b = dt.strftime("%m-%d")
            t_entry = agg_time.setdefault(b, {
                "bucket": b,
                "call_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_krw": 0.0,
                "cost_usd": 0.0,
            })
            t_entry["call_count"] += 1
            t_entry["input_tokens"] += lg["input_tokens"]
            t_entry["output_tokens"] += lg["output_tokens"]
            t_entry["total_tokens"] += lg["input_tokens"] + lg["output_tokens"]
            t_entry["cost_krw"] += lg["cost_krw"]
            t_entry["cost_usd"] += lg["cost_usd"]
        timeline = sorted(agg_time.values(), key=lambda x: x["bucket"], reverse=True)[:10]

        def _sum_logs(logs_subset):
            tok = sum(x["input_tokens"] + x["output_tokens"] for x in logs_subset)
            krw = round(sum(x["cost_krw"] for x in logs_subset), 4)
            usd = round(sum(x["cost_usd"] for x in logs_subset), 6)
            return {"tokens": tok, "cost_krw": krw, "cost_usd": usd}

        period_summary = {
            "daily": _sum_logs([x for x in MEMORY_USAGE_LOGS if x["created_at"] >= today_start]),
            "weekly": _sum_logs([x for x in MEMORY_USAGE_LOGS if x["created_at"] >= week_start]),
            "monthly": _sum_logs([x for x in MEMORY_USAGE_LOGS if x["created_at"] >= month_start]),
            "all": _sum_logs(MEMORY_USAGE_LOGS),
        }

    # 모든 지원 모델 목록을 기본 포함하여 0토큰 모델도 확인 가능하게 구성
    existing_keys = {m["model_key"] for m in by_model}
    for mk, pinfo in MODEL_PRICING.items():
        if mk in ("gemini-2.5-pro",):
            continue
        if mk not in existing_keys:
            by_model.append({
                "model_key": mk,
                "model_label": pinfo["label"],
                "call_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "cost_krw": 0.0,
            })

    return {
        "period": period,
        "period_summary": period_summary,
        "by_model": by_model,
        "timeline": timeline,
        "pricing": MODEL_PRICING,
        "usd_to_krw": USD_TO_KRW,
    }


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_storage()
    yield
    if DB_POOL is not None:
        DB_POOL.close()


# ==============================================================================
# [1] FastAPI 앱 초기화
# ==============================================================================
app = FastAPI(
    title="AI 하이브리드 챗봇 API",
    description="Google Vertex AI (ADC) 및 로컬 Ollama 연동 챗봇 서비스",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """서버 내 모든 예외를 안전하게 JSON 형태로 반환합니다."""
    import traceback
    trace_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    print(f"❌ [서버 내부 예외 발생]: {exc}\n{trace_text}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"서버 내부 오류: {str(exc)}", "error_type": type(exc).__name__},
    )

# 정적 파일(CSS, JS) 마운트
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# ==============================================================================
# [2] 모델 초기화 및 도구 설정 함수
# ==============================================================================
def get_vertex_model(
    enable_google_search: bool = False,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
    reasoning_effort: str | None = "medium",
):
    """
    GCP ADC 인증을 거친 Vertex AI Gemini 모델 인스턴스를 반환합니다.
    - enable_google_search: True일 경우 제미나이의 실시간 구글 검색(Google Search Grounding) 활성화
    - max_output_tokens: 최대 출력 토큰 수 제한
    - temperature: 답변 생성 다양성 (0.0 ~ 2.0)
    - reasoning_effort: 추론/사고 수준 ("off", "low", "medium", "high") -> thinking_budget 연동
    """
    credentials, default_project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip() or default_project
    if not project_id:
        raise ValueError(
            "GCP 프로젝트 ID를 찾을 수 없습니다. "
            "`gcloud config set project [프로젝트ID]`를 먼저 실행하세요."
        )

    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    raw_model_id = os.getenv("GEMINI_MODEL", "").strip()
    if not raw_model_id or "3.8" in raw_model_id:
        model_id = "gemini-2.5-flash"
    else:
        model_id = raw_model_id

    # reasoning_effort를 Vertex AI Gemini thinking_budget으로 매핑
    budget_map = {
        "off": 0,
        "none": 0,
        "low": 1024,
        "medium": 2048,
        "high": 8192,
    }
    effort_key = (reasoning_effort or "medium").lower()
    thinking_budget = budget_map.get(effort_key, 2048)

    base_model = ChatGoogleGenerativeAI(
        model=model_id,
        vertexai=True,
        credentials=credentials,
        project=project_id,
        location=location,
        streaming=False,
        timeout=120,
        max_retries=3,
        max_output_tokens=max_output_tokens,
        temperature=temperature if temperature is not None else 0.7,
        thinking_budget=thinking_budget,
    )

    # 구글 검색(Google Search Grounding) 활성화 시 bind_tools를 통해 정식 Google Search Tool 연동
    if enable_google_search:
        model = base_model.bind_tools([{"google_search": {}}])
    else:
        model = base_model

    return model, project_id, location, model_id


def get_ollama_model(
    max_output_tokens: int | None = None,
    temperature: float | None = None,
):
    """로컬 Ollama에서 실행되는 gemma4:e2b 모델 인스턴스를 반환합니다."""
    return ChatOllama(
        model="gemma4:e2b",
        temperature=temperature if temperature is not None else 0.7,
        num_predict=max_output_tokens,
    )


# ==============================================================================
# [2-0] Code Execution (Python 코드 실행) 도구: execute_python_code
# ==============================================================================
@tool
def execute_python_code(code: str) -> str:
    """Python 코드를 실행하고 표준 출력(stdout) 결과 또는 에러 메시지를 반환합니다.
    수학 계산, 데이터 분석, 통계, 문자열 처리, 알고리즘 구현, 시뮬레이션 등에 사용하세요.
    결과를 확인하려면 코드 내에서 반드시 print(...)를 호출해야 표준 출력으로 결과를 얻을 수 있습니다.
    code: 실행할 완전하고 독립적인 Python 코드 문자열
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=15,
            encoding="utf-8",
            errors="replace",
        )
        out = proc.stdout.strip()
        err = proc.stderr.strip()
        if proc.returncode == 0:
            return out if out else "(코드 실행 완료: 출력된 결과가 없습니다)"
        else:
            return f"실행 에러 (종료코드 {proc.returncode}):\n{err}"
    except subprocess.TimeoutExpired:
        return "실행 에러: 코드 실행 시간 초과 (15초 초과)"
    except Exception as e:
        return f"실행 실패: {e}"


# ==============================================================================
# [2-1] Programmable Search Engine (Custom Search JSON API) 도구: google_cse_search
# 필요한 환경변수:
#   GOOGLE_CSE_API_KEY : Custom Search JSON API 용 API 키 (GCP 콘솔에서 발급)
#   GOOGLE_CSE_ID      : Programmable Search Engine ID (cx 값)
# ==============================================================================
@tool(response_format="content_and_artifact")
def google_cse_search(query: str) -> tuple[str, list[dict]]:
    """Google Programmable Search Engine(Custom Search JSON API)으로 웹을 검색합니다.
    최신 뉴스, 날씨, 실시간 정보 등 모델이 모르는 내용을 찾을 때 사용하세요.
    LangGraph Store를 통해 검색 결과를 캐싱하여 API 호출 비용과 지연시간을 획기적으로 줄입니다.
    query: 검색어 (예: '서울 문정동 오늘 날씨')
    """
    global CACHE_STATS
    CACHE_STATS["total_calls"] += 1
    clean_query = query.strip().lower()

    # 1. LangGraph Store 캐시 확인
    try:
        cached_item = GLOBAL_STORE.get(SEARCH_CACHE_NAMESPACE, clean_query)
        if cached_item and cached_item.value:
            val = cached_item.value
            cached_time = val.get("timestamp", 0)
            elapsed = time.time() - cached_time
            if elapsed < SEARCH_CACHE_TTL:
                CACHE_STATS["hits"] += 1
                sources = val.get("sources", [])
                cached_sources = [{**s, "cached": True} for s in sources]
                cached_header = f"[⚡ LangGraph Store 검색 캐시 HIT ({int(elapsed)}초 전 검색됨)]\n"
                return cached_header + val.get("text", ""), cached_sources
    except Exception as e:
        print(f"캐시 조회 경고 (실시간 검색 진행): {e}")

    # 2. 캐시 미스 (실시간 검색 API 호출)
    CACHE_STATS["misses"] += 1
    api_key = os.getenv("GOOGLE_CSE_API_KEY", "").strip()
    cse_id = os.getenv("GOOGLE_CSE_ID", "").strip()
    if not api_key or not cse_id:
        return "검색 실패: GOOGLE_CSE_API_KEY / GOOGLE_CSE_ID 환경변수가 설정되지 않았습니다.", []

    try:
        resp = httpx.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"key": api_key, "cx": cse_id, "q": query, "num": 5, "hl": "ko"},
            timeout=15,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
    except Exception as e:
        return f"검색 실패: {e}", []

    if not items:
        return "검색 결과가 없습니다.", []

    sources = [{"title": it.get("title", ""), "uri": it.get("link", ""), "cached": False} for it in items]
    lines = [
        f"[{i}] {it.get('title', '')}\n{it.get('snippet', '')}\n{it.get('link', '')}"
        for i, it in enumerate(items, 1)
    ]
    result_text = "\n\n".join(lines)

    # 3. LangGraph Store에 결과 저장
    try:
        GLOBAL_STORE.put(
            SEARCH_CACHE_NAMESPACE,
            clean_query,
            {
                "query": query,
                "text": result_text,
                "sources": sources,
                "timestamp": time.time(),
                "cached_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
    except Exception as e:
        print(f"캐시 저장 경고: {e}")

    return result_text, sources


# ==============================================================================
# [2-2] 실시간 날씨 조회 및 그래픽 위젯 데이터 생성 (Open-Meteo 연동)
# ==============================================================================
WMO_WEATHER_MAP = {
    0: ("맑음", "sunny"),
    1: ("대체로 맑음", "sunny"),
    2: ("구름 조금", "partly_cloudy"),
    3: ("흐림", "cloudy"),
    45: ("안개", "cloudy"),
    48: ("짙은 안개", "cloudy"),
    51: ("약한 이슬비", "rain"),
    53: ("이슬비", "rain"),
    55: ("강한 이슬비", "rain"),
    61: ("약한 비", "rain"),
    63: ("비", "rain"),
    65: ("강한 비", "rain"),
    71: ("약한 눈", "snow"),
    73: ("눈", "snow"),
    75: ("강한 눈", "snow"),
    77: ("싸락눈", "snow"),
    80: ("소나기", "rain"),
    81: ("강한 소나기", "rain"),
    82: ("폭우", "rain"),
    85: ("눈보라", "snow"),
    86: ("강한 눈보라", "snow"),
    95: ("뇌우", "thunder"),
    96: ("우박 동반 뇌우", "thunder"),
    99: ("강한 뇌우", "thunder"),
}

FALLBACK_COORDS = {
    "문정동": ("서울특별시 송파구 문정동", 37.4878, 127.1225),
    "송파구": ("서울특별시 송파구", 37.5145, 127.1059),
    "강남": ("서울특별시 강남구", 37.5172, 127.0473),
    "강남구": ("서울특별시 강남구", 37.5172, 127.0473),
    "서울": ("서울특별시", 37.5665, 126.9780),
    "부산": ("부산광역시", 35.1796, 129.0756),
    "인천": ("인천광역시", 37.4563, 126.7052),
    "대구": ("대구광역시", 35.8714, 128.6014),
    "대전": ("대전광역시", 36.3504, 127.3845),
    "광주": ("광주광역시", 35.1595, 126.8526),
    "울산": ("울산광역시", 35.5384, 129.3114),
    "세종": ("세종특별자치시", 36.4800, 127.2890),
    "수원": ("경기도 수원시", 37.2636, 127.0286),
    "성남": ("경기도 성남시", 37.4200, 127.1267),
    "분당": ("경기도 성남시 분당구", 37.3827, 127.1189),
    "판교": ("경기도 성남시 분당구 판교동", 37.3948, 127.1112),
    "고양": ("경기도 고양시", 37.6584, 126.8320),
    "일산": ("경기도 고양시 일산", 37.6584, 126.7700),
    "용인": ("경기도 용인시", 37.2411, 127.1776),
    "제주": ("제주특별자치도 제주시", 33.4996, 126.5312),
    "서귀포": ("제주특별자치도 서귀포시", 33.2541, 126.5601),
    "강릉": ("강원특별자치도 강릉시", 37.7519, 128.8761),
    "춘천": ("강원특별자치도 춘천시", 37.8813, 127.7298),
    "전주": ("전북특별자치도 전주시", 35.8242, 127.1480),
    "청주": ("충청북도 청주시", 36.6424, 127.4890),
    "천안": ("충청남도 천안시", 36.8151, 127.1139),
    "포항": ("경상북도 포항시", 36.0190, 129.3435),
    "창원": ("경상남도 창원시", 35.2280, 128.6811),
    "여수": ("전라남도 여수시", 34.7604, 127.6622),
}

WEEKDAY_KO = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]


def _wmo_to_info(code: int, is_day: int = 1) -> tuple[str, str]:
    cond_text, icon_key = WMO_WEATHER_MAP.get(int(code), ("구름 조금", "partly_cloudy"))
    if icon_key == "sunny" and not is_day:
        icon_key = "clear_night"
    elif icon_key == "partly_cloudy" and not is_day:
        icon_key = "cloudy_night"
    return cond_text, icon_key


def extract_weather_location(user_msg: str) -> str | None:
    """사용자 질문이 날씨 관련인지 판별하고 지역명을 추출합니다."""
    weather_keywords = ["날씨", "기온", "온도", "비 오", "비오", "눈 오", "눈오", "우산", "일기예보", "강수", "미세먼지", "추워", "더워"]
    if not any(k in user_msg for k in weather_keywords):
        return None

    for key in FALLBACK_COORDS:
        if key in user_msg:
            return key

    stop_words = {
        "오늘", "내일", "모레", "이번주", "주말", "지금", "현재", "실시간", "날씨", "기온", "온도",
        "어때", "어때요", "알려줘", "알려주세요", "검색", "검색해줘", "어떨까", "비", "눈", "강수확률", "우산", "있어", "있나요"
    }
    tokens = re.findall(r"[가-힣A-Za-z]{2,}", user_msg)
    candidates = []
    for t in tokens:
        clean_t = re.sub(r"(의|은|는|이|가|에서|에|도|를|을|랑|하고)$", "", t)
        if clean_t and clean_t not in stop_words and not any(k in clean_t for k in ["날씨", "기온", "알려", "어때"]):
            candidates.append(clean_t)

    return candidates[-1] if candidates else "서울"


def fetch_weather_widget_data(location_query: str) -> dict | None:
    """지역명을 기반으로 Open-Meteo API를 호출하여 Gemini 웹 스타일 날씨 위젯 데이터를 생성합니다."""
    loc_q = (location_query or "서울").strip()
    display_location = loc_q
    lat, lon = None, None

    if loc_q in FALLBACK_COORDS:
        display_location, lat, lon = FALLBACK_COORDS[loc_q]
    else:
        for k, (disp, f_lat, f_lon) in FALLBACK_COORDS.items():
            if k in loc_q or loc_q in k:
                display_location, lat, lon = disp, f_lat, f_lon
                break

    # Open-Meteo Geocoding API 조회
    try:
        geo_resp = httpx.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": loc_q, "count": 5, "language": "ko"},
            timeout=6,
        )
        if geo_resp.status_code == 200:
            results = geo_resp.json().get("results") or []
            kr_results = [r for r in results if r.get("country_code") == "KR"]
            chosen = kr_results[0] if kr_results else (results[0] if results else None)
            if chosen:
                lat = chosen["latitude"]
                lon = chosen["longitude"]
                parts = []
                for field in ("admin1", "admin2", "admin3", "name"):
                    val = (chosen.get(field) or "").strip()
                    if val and re.search(r"[가-힣]", val) and val not in parts:
                        parts.append(val)
                if parts:
                    # 문정1동 -> 문정동 등 사용자가 입력한 동 이름 자연스럽게 보정
                    joined = " ".join(parts)
                    if loc_q.endswith("동") and loc_q not in joined:
                        parts[-1] = loc_q
                        joined = " ".join(parts)
                    display_location = joined
    except Exception as e:
        print(f"Geocoding 조회 경고 ({loc_q}): {e}")

    if lat is None or lon is None:
        display_location, lat, lon = FALLBACK_COORDS["서울"]

    try:
        fc_resp = httpx.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,weather_code,wind_speed_10m,precipitation,relative_humidity_2m,is_day",
                "hourly": "temperature_2m,precipitation_probability,precipitation,weather_code,wind_speed_10m,is_day",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": "Asia/Seoul",
                "forecast_days": 2,
            },
            timeout=8,
        )
        fc_resp.raise_for_status()
        data = fc_resp.json()
    except Exception as e:
        print(f"Open-Meteo 날씨 조회 실패 ({loc_q}): {e}")
        return None

    now = datetime.now()
    day_name = WEEKDAY_KO[now.weekday()]

    current = data.get("current", {})
    daily = data.get("daily", {})
    hourly = data.get("hourly", {})

    curr_temp_c = round(current.get("temperature_2m", 20))
    curr_code = current.get("weather_code", 0)
    curr_is_day = current.get("is_day", 1)
    condition_text, current_icon = _wmo_to_info(curr_code, curr_is_day)

    high_list = daily.get("temperature_2m_max") or [curr_temp_c]
    low_list = daily.get("temperature_2m_min") or [curr_temp_c]
    pop_max_list = daily.get("precipitation_probability_max") or [0]

    high_c = round(high_list[0])
    low_c = round(low_list[0])
    precip_prob = int(pop_max_list[0] or 0)

    # 현재 시각 이후 8개 시간대 추출
    times = hourly.get("time") or []
    h_temps = hourly.get("temperature_2m") or []
    h_pops = hourly.get("precipitation_probability") or []
    h_precips = hourly.get("precipitation") or []
    h_codes = hourly.get("weather_code") or []
    h_winds = hourly.get("wind_speed_10m") or []
    h_is_days = hourly.get("is_day") or []

    now_str = now.strftime("%Y-%m-%dT%H:00")
    start_idx = 0
    for idx, t_str in enumerate(times):
        if t_str >= now_str:
            start_idx = idx
            break

    hourly_items = []
    for i in range(start_idx, min(start_idx + 8, len(times))):
        t_str = times[i]
        try:
            dt = datetime.fromisoformat(t_str)
            h = dt.hour
        except Exception:
            h = (now.hour + (i - start_idx)) % 24

        ampm = "오전" if h < 12 else "오후"
        disp_h = h if 1 <= h <= 12 else (h - 12 if h > 12 else 12)
        time_label = f"{disp_h} {ampm}"

        t_c = round(h_temps[i]) if i < len(h_temps) and h_temps[i] is not None else curr_temp_c
        pop = int(h_pops[i] or 0) if i < len(h_pops) else 0
        precip_mm = round(float(h_precips[i] or 0.0), 1) if i < len(h_precips) else 0.0
        w_code = int(h_codes[i] or 0) if i < len(h_codes) else 0
        wind_ms = round(float(h_winds[i] or 0.0) / 3.6, 1) if i < len(h_winds) else 0.0
        is_d = int(h_is_days[i]) if i < len(h_is_days) and h_is_days[i] is not None else (1 if 6 <= h < 19 else 0)

        h_cond, h_icon = _wmo_to_info(w_code, is_d)
        hourly_items.append({
            "time_label": time_label,
            "icon": h_icon,
            "condition": h_cond,
            "precip_prob": pop,
            "precip_mm": precip_mm,
            "temp_c": t_c,
            "temp_f": round(t_c * 9 / 5 + 32),
            "wind_ms": wind_ms,
        })

    return {
        "location": display_location,
        "day_of_week": day_name,
        "current_temp_c": curr_temp_c,
        "current_temp_f": round(curr_temp_c * 9 / 5 + 32),
        "condition": condition_text,
        "icon": current_icon,
        "high_c": high_c,
        "high_f": round(high_c * 9 / 5 + 32),
        "low_c": low_c,
        "low_f": round(low_c * 9 / 5 + 32),
        "precip_prob": precip_prob,
        "humidity": int(current.get("relative_humidity_2m") or 0),
        "wind_ms": round(float(current.get("wind_speed_10m") or 0.0) / 3.6, 1),
        "hourly": hourly_items,
        "provider": "웨더뉴스 · Open-Meteo",
    }


def fetch_url_article(url: str) -> dict | None:
    """웹 URL에서 기사/페이지 본문과 제목, 기자/작성자를 추출합니다."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        }
        with httpx.Client(timeout=12, follow_redirects=True, headers=headers) as client:
            resp = client.get(url)
            if resp.status_code >= 400:
                return None
            html = resp.text

        soup = BeautifulSoup(html, "html.parser")

        # 1. 제목 탐색
        title = ""
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = og_title["content"].strip()
        elif soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)

        # 2. 기자/작성자 탐색
        author = ""
        for meta_name in ["author", "article:author", "dable:author", "byline", "writer"]:
            m = soup.find("meta", attrs={"name": meta_name}) or soup.find("meta", attrs={"property": meta_name})
            if m and m.get("content"):
                author = m["content"].strip()
                break

        if not author:
            for elem in soup.find_all(["span", "p", "div", "a", "strong"]):
                t = elem.get_text(strip=True)
                if "기자" in t and 2 <= len(t) <= 25 and not any(k in t for k in ["구독", "제보", "댓글", "다른 기사"]):
                    author = t
                    break

        # 3. 불필요 태그 제거 후 본문 텍스트 추출
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
            tag.decompose()

        article = (
            soup.find("article")
            or soup.find("main")
            or soup.find(class_=lambda x: x and any(k in str(x).lower() for k in ["article-body", "article_body", "news_body", "article_text", "article-content", "content"]))
        )
        target = article if article else soup
        raw_text = target.get_text(separator="\n", strip=True)
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        body_text = "\n".join(lines)[:4000]

        return {
            "url": url,
            "title": title,
            "author": author,
            "text": body_text,
        }
    except Exception as e:
        print(f"URL 수집 실패 ({url}): {e}")
        return None


def format_message_content(content) -> str:
    """Gemini 등 멀티모달 모델이 반환하는 content 블록 리스트에서 순수 텍스트만 추출합니다."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
            elif hasattr(part, "text"):
                parts.append(part.text)
        return "\n".join(parts) if parts else ""
    return str(content) if content is not None else ""


def extract_bot_reply(message) -> tuple[str, list[dict]]:
    """
    AIMessage 객체에서 순수 텍스트 답변과 구글 검색 출처 목록을 분리 추출합니다.
    반환값: (reply_text, sources_list)
    """
    text = ""
    if hasattr(message, "text") and message.text:
        text = str(message.text)
    elif hasattr(message, "content"):
        text = format_message_content(message.content)
    elif isinstance(message, str):
        text = message
    else:
        text = format_message_content(message)

    if not text.strip() and hasattr(message, "tool_calls") and message.tool_calls:
        calls = [tc.get("name", "tool") for tc in message.tool_calls]
        text = f"도구 호출을 시도했으나 결과를 불러오지 못했습니다 ({', '.join(calls)})."

    if not text.strip():
        text = "답변을 불러오지 못했습니다. 잠시 후 다시 질문해 주세요."

    # 구글 검색 그라운딩 출처 파싱
    sources = []
    metadata = getattr(message, "response_metadata", {}) or {}
    grounding = metadata.get("grounding_metadata", {})
    if grounding:
        chunks = grounding.get("grounding_chunks", []) or []
        seen = set()
        for chunk in chunks:
            web = chunk.get("web", {}) if isinstance(chunk, dict) else {}
            title = web.get("title", "").strip()
            uri = web.get("uri", "").strip()
            if uri and uri not in seen:
                seen.add(uri)
                sources.append({"title": title or uri, "uri": uri})

    return text, sources


def _get_turn_messages(response_messages):
    """전체 메시지 중 이번 턴(마지막 사용자 메시지 이후)의 메시지만 반환합니다."""
    last_human = 0
    for i, m in enumerate(response_messages):
        if getattr(m, "type", "") == "human":
            last_human = i
    return response_messages[last_human:]


def collect_cse_sources(response_messages) -> list[dict]:
    """이번 턴에 실행된 google_cse_search 도구의 검색 출처(artifact)를 모아 반환합니다."""
    sources, seen = [], set()
    for m in _get_turn_messages(response_messages):
        if getattr(m, "type", "") == "tool" and isinstance(getattr(m, "artifact", None), list):
            for s in m.artifact:
                if s.get("uri") and s["uri"] not in seen:
                    seen.add(s["uri"])
                    sources.append(s)
    return sources


def collect_code_executions(response_messages) -> list[dict]:
    """이번 턴에 실행된 Python 코드 실행 내역(코드 및 실행 결과)을 수집합니다."""
    executions = []
    turn_msgs = _get_turn_messages(response_messages)

    tool_call_map = {}
    for m in turn_msgs:
        if getattr(m, "type", "") == "ai" and hasattr(m, "tool_calls") and m.tool_calls:
            for tc in m.tool_calls:
                tname = tc.get("name", "")
                if tname in ("execute_python_code", "python_interpreter", "code_execution"):
                    t_id = tc.get("id", "")
                    code_val = tc.get("args", {}).get("code", "")
                    if t_id and code_val:
                        tool_call_map[t_id] = code_val

    for m in turn_msgs:
        if getattr(m, "type", "") == "tool":
            t_id = getattr(m, "tool_call_id", "")
            if t_id in tool_call_map:
                executions.append({
                    "code": tool_call_map[t_id],
                    "output": format_message_content(getattr(m, "content", "")),
                })

    return executions


def calculate_token_usage_and_cost(response_messages, model_choice: str, reasoning_effort: str = "medium"):
    """
    LangChain 응답 메시지에서 토큰 사용량을 추출하고 모델별 단가에 따라 예상 비용을 계산합니다.
    (검색 도구 호출로 모델이 여러 번 호출될 수 있으므로 이번 턴의 AI 메시지 사용량을 합산합니다.)
    """
    input_tokens = 0
    output_tokens = 0
    turn_msgs = _get_turn_messages(response_messages)

    for msg in turn_msgs:
        if getattr(msg, "type", "") != "ai":
            continue
        usage = getattr(msg, "usage_metadata", None)
        if isinstance(usage, dict):
            input_tokens += usage.get("input_tokens", 0)
            output_tokens += usage.get("output_tokens", 0)
        elif isinstance(getattr(msg, "response_metadata", None), dict):
            meta = msg.response_metadata
            if "prompt_eval_count" in meta or "eval_count" in meta:
                input_tokens += meta.get("prompt_eval_count", 0)
                output_tokens += meta.get("eval_count", 0)

    if input_tokens == 0 and output_tokens == 0:
        last_msg = response_messages[-1]
        user_text = turn_msgs[0].content if turn_msgs else ""
        assistant_text = format_message_content(last_msg.content)
        input_tokens = max(1, len(str(user_text)) // 3)
        output_tokens = max(1, len(assistant_text) // 3)

    model_key, model_label = resolve_model_key_and_label(model_choice, reasoning_effort)
    pricing = MODEL_PRICING.get(model_key, MODEL_PRICING["gemini-2.5-flash"])

    cost_usd = (
        (input_tokens / 1_000_000.0) * pricing["input_per_1m"]
        + (output_tokens / 1_000_000.0) * pricing["output_per_1m"]
    )
    cost_krw = cost_usd * USD_TO_KRW

    return {
        "model_key": model_key,
        "model_label": model_label,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
        "cost_krw": cost_krw,
    }


# ==============================================================================
# [3] Pydantic 요청/응답 모델 정의
# ==============================================================================
class ChatRequest(BaseModel):
    message: str
    model: str = "gemini"  # "gemini" 또는 "ollama"
    # 검색 모드: "off"(끔) | "grounding"(Vertex AI Grounding with Google Search, Gemini 전용)
    #           | "cse"(Programmable Search Engine Custom Search JSON API, google_cse_search 도구)
    search_mode: str = "grounding"
    thread_id: str | None = None
    max_output_tokens: int | None = 2048
    temperature: float | None = 0.7
    # 추론 수준 (Reasoning Effort): "off" | "low" | "medium" | "high"
    reasoning_effort: str = "medium"
    # 구조화된 출력 (Structured Output) 원샷 예시 설정
    structured_output: bool = False
    one_shot_input: str | None = None
    one_shot_output: str | None = None
    # Code execution (파이썬 코드 실행) 툴 활성화 여부
    code_execution: bool = False


# ==============================================================================
# [4] REST API 엔드포인트
# ==============================================================================
@app.get("/api/health")
async def health():
    """서버 상태 및 기본 모델/환경변수 정보를 반환합니다."""
    proj = os.getenv("GOOGLE_CLOUD_PROJECT", "iceu-songpa04")
    loc = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    return {
        "status": "healthy",
        "default_gemini_model": "gemini-2.5-flash",
        "default_ollama_model": "gemma4:e2b",
        "gcp_project": proj,
        "gcp_location": loc,
        "cse_configured": bool(os.getenv("GOOGLE_CSE_API_KEY") and os.getenv("GOOGLE_CSE_ID")),
        "storage_backend": STORAGE_BACKEND,
    }


@app.post("/api/reset")
def reset_session():
    """새로운 thread_id를 발급(DB에 새 세션 행 생성)하여 새 대화를 시작합니다. 이전 대화는 DB에 그대로 남습니다."""
    new_thread_id = str(uuid.uuid4())[:8]
    create_session(new_thread_id)
    return {
        "status": "ok",
        "thread_id": new_thread_id,
        "message": "새로운 대화 세션이 시작되었습니다.",
    }


@app.get("/api/sessions")
def get_sessions():
    """저장된 전체 대화 세션 목록을 반환합니다."""
    sessions = list_sessions()
    for s in sessions:
        if s.get("updated_at"):
            s["updated_at"] = str(s["updated_at"])
    return {"sessions": sessions}


class RenameSessionRequest(BaseModel):
    title: str


@app.patch("/api/sessions/{thread_id}")
def rename_session_endpoint(thread_id: str, req: RenameSessionRequest):
    """세션의 제목(이름)을 변경합니다."""
    new_title = req.title.strip()
    if not new_title:
        raise HTTPException(status_code=400, detail="변경할 대화 제목을 입력하세요.")
    update_session_title(thread_id, new_title)
    return {"status": "ok", "thread_id": thread_id, "title": new_title}


@app.delete("/api/sessions/{thread_id}")
def delete_session_endpoint(thread_id: str):
    """세션을 삭제합니다."""
    delete_session(thread_id)
    return {"status": "ok", "message": f"세션 {thread_id}이 삭제되었습니다."}


def _load_messages(thread_id: str) -> list[dict]:
    """체크포인터(PostgreSQL)에 저장된 대화 중 사용자/AI 텍스트 메시지만 꺼냅니다."""
    config = {"configurable": {"thread_id": thread_id}}
    tup = CHECKPOINTER.get_tuple(config)
    raw = tup.checkpoint.get("channel_values", {}).get("messages", []) if tup else []
    messages = []
    for m in raw:
        m_type = getattr(m, "type", "unknown")
        text = format_message_content(getattr(m, "content", ""))
        if m_type in ("human", "ai") and text.strip():
            messages.append({"type": m_type, "content": text})
    return messages


@app.get("/api/history/{thread_id}")
def get_history(thread_id: str):
    """세션(thread_id)에 저장된 대화 메시지를 조회합니다."""
    messages = _load_messages(thread_id)
    return {"thread_id": thread_id, "count": len(messages), "messages": messages}


@app.get("/api/session/{thread_id}")
def get_session(thread_id: str):
    """페이지 새로고침/서버 재시작 후 세션을 복원하기 위한 대화 내용과 토큰 통계를 반환합니다."""
    messages = _load_messages(thread_id)
    stats = get_session_stats(thread_id)
    return {
        "thread_id": thread_id,
        "messages": messages,
        "token_stats": {
            "total_tokens": stats["total_input"] + stats["total_output"],
            "total_cost_krw": round(stats["total_cost_krw"], 4),
            "total_cost_usd": round(stats["total_cost_usd"], 6),
        },
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    """
    사용자의 질문을 받아 LangChain Agent를 호출하고 답변, 검색 출처, 토큰 사용량을 반환합니다.
    """
    user_msg = req.message.strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="메시지를 입력하세요.")

    thread_id = req.thread_id or str(uuid.uuid4())[:8]
    create_session(thread_id)
    update_session_title_if_empty(thread_id, user_msg)

    # 1. 모델 준비 (search_mode == "grounding"일 때만 Vertex AI Grounding 도구 바인딩)
    search_mode = req.search_mode if req.search_mode in ("off", "grounding", "cse") else "grounding"
    use_grounding = search_mode == "grounding" and req.model == "gemini"
    use_cse = search_mode == "cse"

    if use_cse and not (os.getenv("GOOGLE_CSE_API_KEY") and os.getenv("GOOGLE_CSE_ID")):
        raise HTTPException(
            status_code=400,
            detail="Custom Search를 쓰려면 .env에 GOOGLE_CSE_API_KEY, GOOGLE_CSE_ID를 설정하세요.",
        )

    try:
        if req.model == "gemini":
            model, project_id, location, model_id = get_vertex_model(
                enable_google_search=use_grounding,
                max_output_tokens=req.max_output_tokens,
                temperature=req.temperature,
                reasoning_effort=req.reasoning_effort,
            )
        else:
            model = get_ollama_model(
                max_output_tokens=req.max_output_tokens,
                temperature=req.temperature,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI 모델 초기화 실패: {e}")

    # 2. 검색 모드에 따른 시스템 프롬프트 및 도구 구성
    base_prompt = "당신은 친절하고 전문적인 AI 챗봇 어시스턴트입니다. "
    tools = []
    if use_grounding:
        system_prompt = base_prompt + (
            "구글 검색 기능을 활용하여 최신 정보나 실시간 데이터(뉴스, 날씨, 시세 등)를 직접 검색하여 "
            "정확하고 신뢰할 수 있는 정보를 한국어로 답변하세요. 사용자의 이전 대화 맥락도 잘 기억하세요."
        )
    elif use_cse:
        system_prompt = base_prompt + (
            "최신 정보나 실시간 데이터가 필요하면 google_cse_search 도구로 검색한 뒤, "
            "검색 결과를 근거로 한국어로 답변하세요. 사용자의 이전 대화 맥락도 잘 기억하세요."
        )
        tools.append(google_cse_search)
    else:
        system_prompt = base_prompt + (
            "사용자의 이전 대화 맥락을 잘 기억하여 자연스럽고 도움이 되는 한국어로 답변하세요."
        )

    # Code Execution (파이썬 코드 실행) 도구 활성화
    if req.code_execution:
        tools.append(execute_python_code)
        system_prompt += (
            "\n\n[★ 파이썬 코드 실행 (Code Execution) 도구 사용 지침]\n"
            "당신에게는 `execute_python_code` 도구가 제공됩니다.\n"
            "수학 계산, 통계, 데이터 분석, 복잡한 공식, 시뮬레이션, 알고리즘 구현 등이 필요한 경우 "
            "머릿속으로 어림짐작하지 말고 반드시 완전한 Python 코드를 작성하여 `execute_python_code` 도구를 실행하세요.\n"
            "코드 실행 결과(표준 출력 stdout)를 반드시 확인한 후, 그 실제 계산/실행 결과를 바탕으로 사용자에게 친절하게 설명하여 최종 답변하세요.\n"
            "코드 내에서 보고자 하는 결과값은 반드시 `print(...)`를 사용하여 출력해야 결과를 얻을 수 있습니다.\n"
        )

    # 구조화된 출력 (Structured Output - One-Shot 예시 포맷팅 반영: 일반 텍스트 및 JSON 모두 지원)
    if req.structured_output and req.one_shot_output and req.one_shot_output.strip():
        one_shot_format = req.one_shot_output.strip()
        structured_guide = (
            "\n\n[★ 필수 규칙: 구조화된 출력 (Structured Output)]\n"
            "당신은 반드시 아래에 제시된 [원샷(One-Shot) 출력 포맷 예시]의 서식, 라벨, 레이아웃을 그대로 준수하여 답변해야 합니다.\n"
            "인삿말, 서론/결론, 마크다운 꾸밈 등의 불필요한 미사여구를 일절 추가하지 마세요.\n"
            "오직 제시된 예시 포맷의 각 항목 라벨(예: '뉴스 제목 :', '기자 이름 :', '뉴스 내용 요약 :' 등)에 맞춰 내용을 채워 출력하세요.\n"
            "라벨 뒤에 콜론(:)이 있는 경우, 반드시 해당 라벨 바로 뒤에 분석/요약된 내용을 기재하세요.\n"
        )
        if req.one_shot_input and req.one_shot_input.strip():
            structured_guide += f"- [참고용 원샷 예시 입력]: {req.one_shot_input.strip()}\n"
        structured_guide += f"- [반드시 준수할 원샷 출력 포맷]:\n{one_shot_format}\n"
        system_prompt += structured_guide

    # 3. URL 감지 및 웹페이지 본문 자동 수집 (URL Context 연동)
    urls = re.findall(r"https?://[^\s<>\"']+", user_msg)
    url_sources = []
    agent_input_msg = user_msg
    if urls:
        web_blocks = []
        for u in urls[:2]:
            article_info = fetch_url_article(u)
            if article_info:
                url_sources.append({
                    "title": article_info["title"] or u,
                    "uri": u,
                    "cached": False,
                })
                block = f"[웹페이지 정보 (URL: {u})]\n- 제목: {article_info['title']}"
                if article_info.get("author"):
                    block += f"\n- 기자/작성자: {article_info['author']}"
                block += f"\n- 본문 내용:\n{article_info['text']}"
                web_blocks.append(block)

        if web_blocks:
            agent_input_msg = (
                f"{user_msg}\n\n"
                f"--- [참고: 입력된 링크 페이지에서 직접 수집한 실제 웹페이지 내용입니다. 이를 근거로 답변하세요] ---\n"
                + "\n\n".join(web_blocks)
            )

    # 3-1. 날씨 질문 감지 및 실시간 날씨 그래픽 위젯 데이터 수집
    weather_data = None
    weather_loc = extract_weather_location(user_msg)
    if weather_loc:
        weather_data = fetch_weather_widget_data(weather_loc)
        if weather_data and not req.structured_output:
            hourly_summary = ", ".join(
                f"{h['time_label']}({h['condition']} {h['temp_c']}°C, 강수 {h['precip_prob']}%)"
                for h in weather_data.get("hourly", [])[:5]
            )
            agent_input_msg = (
                f"{agent_input_msg}\n\n"
                f"--- [실시간 기상 데이터 ({weather_data['location']}) - 화면 상단에 그래픽 날씨 위젯이 함께 출력됩니다] ---\n"
                f"- 현재 기온: {weather_data['current_temp_c']}°C ({weather_data['condition']})\n"
                f"- 최고/최저 기온: 최고 {weather_data['high_c']}°C / 최저 {weather_data['low_c']}°C\n"
                f"- 강수 확률: {weather_data['precip_prob']}% / 풍속: {weather_data['wind_ms']}m/s\n"
                f"- 시간대별 예보: {hourly_summary}\n"
                f"위 기상청/웨더뉴스 데이터를 바탕으로 Gemini 웹 서비스 하단 요약문처럼 1~2문장으로 자연스럽고 간결하게 날씨를 요약해 주세요."
            )

    # 4. LangChain create_agent 생성 (PostgresSaver 체크포인터 & LangGraph Store 연결)
    agent = create_agent(
        model=model,
        tools=tools,
        checkpointer=CHECKPOINTER,
        store=GLOBAL_STORE,
        system_prompt=system_prompt,
    )

    # 5. 에이전트 실행
    config = {"configurable": {"thread_id": thread_id}}
    try:
        response = agent.invoke(
            {"messages": [{"role": "user", "content": agent_input_msg}]},
            config=config,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"답변 생성 실패: {e}")

    last_msg = response["messages"][-1]
    reply_text, sources = extract_bot_reply(last_msg)
    cache_hit = False
    if use_cse:
        sources = collect_cse_sources(response["messages"])
        cache_hit = any(s.get("cached", False) for s in sources)

    # 코드 실행 내역(코드 및 결과) 수집
    code_executions = collect_code_executions(response["messages"])

    if url_sources:
        seen_uris = {s.get("uri") for s in sources}
        for us in url_sources:
            if us["uri"] not in seen_uris:
                sources.insert(0, us)
                seen_uris.add(us["uri"])

    # 5. 토큰 및 비용 통계를 DB(chat_sessions 및 token_usage_logs)에 누적 저장
    turn_stats = calculate_token_usage_and_cost(response["messages"], req.model, req.reasoning_effort)
    sess = add_session_stats(thread_id, turn_stats)
    analytics = get_token_analytics("daily")

    return {
        "reply": reply_text,
        "thread_id": thread_id,
        "sources": sources,
        "weather_data": weather_data,
        "is_structured": req.structured_output,
        "code_executions": code_executions,
        "cache_hit": cache_hit,
        "cache_stats": {
            "hits": CACHE_STATS["hits"],
            "misses": CACHE_STATS["misses"],
            "total_calls": CACHE_STATS["total_calls"],
        },
        "token_stats": {
            "model_key": turn_stats["model_key"],
            "model_label": turn_stats["model_label"],
            "last_input": turn_stats["input_tokens"],
            "last_output": turn_stats["output_tokens"],
            "last_cost_krw": round(turn_stats["cost_krw"], 4),
            "last_cost_usd": round(turn_stats["cost_usd"], 6),
            "total_input": sess["total_input"],
            "total_output": sess["total_output"],
            "total_tokens": sess["total_input"] + sess["total_output"],
            "total_cost_krw": round(sess["total_cost_krw"], 4),
            "total_cost_usd": round(sess["total_cost_usd"], 6),
            "period_summary": analytics["period_summary"],
        },
    }


# ==============================================================================
# [4-0] 모델별 토큰 카운터 및 일별/주간별/월별 사용량 통계 엔드포인트
# ==============================================================================
@app.get("/api/token-stats")
def get_token_stats_endpoint(period: str = "daily"):
    """모델별 누적 토큰 사용량 및 일별(daily)/주간별(weekly)/월별(monthly) 통계를 반환합니다."""
    return get_token_analytics(period)


# ==============================================================================
# [4-1] LangGraph Store 캐시 관리 엔드포인트
# ==============================================================================
@app.get("/api/cache/stats")
def get_cache_stats():
    """LangGraph Store 검색 캐시 통계를 반환합니다."""
    cached_count = 0
    try:
        items = GLOBAL_STORE.search(SEARCH_CACHE_NAMESPACE, limit=100)
        cached_count = len(items)
    except Exception:
        pass
    return {
        "status": "ok",
        "ttl_seconds": SEARCH_CACHE_TTL,
        "ttl_minutes": SEARCH_CACHE_TTL // 60,
        "hits": CACHE_STATS["hits"],
        "misses": CACHE_STATS["misses"],
        "total_calls": CACHE_STATS["total_calls"],
        "cached_count": cached_count,
    }


@app.post("/api/cache/clear")
def clear_cache():
    """LangGraph Store의 검색 캐시를 비웁니다."""
    global CACHE_STATS
    try:
        items = GLOBAL_STORE.search(SEARCH_CACHE_NAMESPACE, limit=100)
        for item in items:
            GLOBAL_STORE.delete(SEARCH_CACHE_NAMESPACE, item.key)
    except Exception as e:
        print(f"캐시 삭제 중 오류: {e}")
    CACHE_STATS = {"hits": 0, "misses": 0, "total_calls": 0}
    return {"status": "ok", "message": "LangGraph 검색 캐시가 초기화되었습니다."}


# ==============================================================================
# [5] 브라우저 웹 UI 제공 엔드포인트
# ==============================================================================
@app.get("/", response_class=FileResponse)
async def serve_ui():
    """웹 브라우저에서 직접 대화할 수 있는 모던 채팅 인터페이스(HTML)를 반환합니다."""
    index_file = BASE_DIR / "templates" / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html 파일을 찾을 수 없습니다.")
    return FileResponse(index_file)


# ==============================================================================
# [6] 메인 실행 엔트리포인트 (python app.py 바로 실행 가능)
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 FastAPI AI 하이브리드 챗봇 서버를 시작합니다.")
    print("👉 웹 브라우저 접속 주소: http://127.0.0.1:8000")
    print("👉 Swagger API 문서: http://127.0.0.1:8000/docs")
    print("=" * 60)
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True, app_dir=str(BASE_DIR))
