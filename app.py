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
import sys
import uuid
import httpx
import uvicorn
import google.auth
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

# .env 파일의 GOOGLE_CSE_API_KEY, GOOGLE_CSE_ID 등을 로드
load_dotenv()

# Windows 콘솔 한글/특수문자 인코딩 보정
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ==============================================================================
# [1] FastAPI 앱 및 전역 메모리 초기화
# ==============================================================================
app = FastAPI(
    title="AI 하이브리드 챗봇 API",
    description="Google Vertex AI (ADC) 및 로컬 Ollama 연동 챗봇 서비스",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 세션별 InMemorySaver 및 토큰/비용 누적 저장소
# LangGraph의 InMemorySaver는 thread_id를 키로 삼아 대화 맥락을 메모리에 보관합니다.
MEMORY_SAVER = InMemorySaver()
SESSION_STATS: dict[str, dict] = {}


# ==============================================================================
# [2] 모델 초기화 및 도구 설정 함수
# ==============================================================================
def get_vertex_model(enable_google_search: bool = False):
    """
    GCP ADC 인증을 거친 Vertex AI Gemini 모델 인스턴스를 반환합니다.
    - enable_google_search: True일 경우 제미나이의 실시간 구글 검색(Google Search Grounding) 활성화
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

    base_model = ChatGoogleGenerativeAI(
        model=model_id,
        vertexai=True,
        credentials=credentials,
        project=project_id,
        location=location,
        streaming=False,
        timeout=120,
        max_retries=3,
    )

    # 구글 검색(Google Search Grounding) 활성화 시 bind_tools를 통해 정식 Google Search Tool 연동
    if enable_google_search:
        model = base_model.bind_tools([{"google_search": {}}])
    else:
        model = base_model

    return model, project_id, location, model_id


def get_ollama_model():
    """로컬 Ollama에서 실행되는 gemma4:e2b 모델 인스턴스를 반환합니다."""
    return ChatOllama(
        model="gemma4:e2b",
        temperature=0,
    )


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
    query: 검색어 (예: '서울 문정동 오늘 날씨')
    """
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

    sources = [{"title": it.get("title", ""), "uri": it.get("link", "")} for it in items]
    lines = [
        f"[{i}] {it.get('title', '')}\n{it.get('snippet', '')}\n{it.get('link', '')}"
        for i, it in enumerate(items, 1)
    ]
    return "\n\n".join(lines), sources


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


def calculate_token_usage_and_cost(response_messages, model_choice: str):
    """
    LangChain 응답 메시지에서 토큰 사용량을 추출하고 예상 비용을 계산합니다.
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

    if "gemini" in model_choice.lower():
        cost_usd = (input_tokens * 0.000000075) + (output_tokens * 0.00000030)
        cost_krw = cost_usd * 1380
    else:
        cost_usd = 0.0
        cost_krw = 0.0

    return {
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
    search_mode: str = "off"
    thread_id: str | None = None


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
    }


@app.post("/api/reset")
async def reset_session():
    """새로운 thread_id를 발급하여 대화 내역 및 토큰 카운터를 초기화합니다."""
    new_thread_id = str(uuid.uuid4())[:8]
    SESSION_STATS[new_thread_id] = {
        "total_input": 0,
        "total_output": 0,
        "total_cost_krw": 0.0,
        "total_cost_usd": 0.0,
    }
    return {
        "status": "ok",
        "thread_id": new_thread_id,
        "message": "새로운 대화 세션이 시작되었습니다.",
    }


@app.get("/api/history/{thread_id}")
async def get_history(thread_id: str):
    """현재 세션(thread_id)의 InMemorySaver에 저장된 실제 LangChain 메시지들을 조회합니다."""
    config = {"configurable": {"thread_id": thread_id}}
    state = MEMORY_SAVER.get(config)
    messages = []
    if state and "channel_values" in state and "messages" in state["channel_values"]:
        for m in state["channel_values"]["messages"]:
            m_type = getattr(m, "type", "unknown")
            text = format_message_content(getattr(m, "content", ""))
            messages.append({"type": m_type, "content": text})
    return {"thread_id": thread_id, "count": len(messages), "messages": messages}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    사용자의 질문을 받아 LangChain Agent를 호출하고 답변, 검색 출처, 토큰 사용량을 반환합니다.
    """
    user_msg = req.message.strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="메시지를 입력하세요.")

    thread_id = req.thread_id or str(uuid.uuid4())[:8]

    # 1. 모델 준비 (search_mode == "grounding"일 때만 Vertex AI Grounding 도구 바인딩)
    search_mode = req.search_mode if req.search_mode in ("off", "grounding", "cse") else "off"
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
                enable_google_search=use_grounding
            )
        else:
            model = get_ollama_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI 모델 초기화 실패: {e}")

    # 2. 검색 모드에 따른 시스템 프롬프트 및 도구 구성
    base_prompt = "당신은 친절하고 전문적인 AI 챗봇 어시스턴트입니다. "
    if use_grounding:
        system_prompt = base_prompt + (
            "구글 검색 기능을 활용하여 최신 정보나 실시간 데이터(뉴스, 날씨, 시세 등)를 직접 검색하여 "
            "정확하고 신뢰할 수 있는 정보를 한국어로 답변하세요. 사용자의 이전 대화 맥락도 잘 기억하세요."
        )
        tools = []
    elif use_cse:
        system_prompt = base_prompt + (
            "최신 정보나 실시간 데이터가 필요하면 google_cse_search 도구로 검색한 뒤, "
            "검색 결과를 근거로 한국어로 답변하세요. 사용자의 이전 대화 맥락도 잘 기억하세요."
        )
        tools = [google_cse_search]
    else:
        system_prompt = base_prompt + (
            "사용자의 이전 대화 맥락을 잘 기억하여 자연스럽고 도움이 되는 한국어로 답변하세요."
        )
        tools = []

    # 3. LangChain create_agent 생성 (InMemorySaver 연결)
    agent = create_agent(
        model=model,
        tools=tools,
        checkpointer=MEMORY_SAVER,
        system_prompt=system_prompt,
    )

    # 4. 에이전트 실행
    config = {"configurable": {"thread_id": thread_id}}
    try:
        response = agent.invoke(
            {"messages": [{"role": "user", "content": user_msg}]},
            config=config,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"답변 생성 실패: {e}")

    last_msg = response["messages"][-1]
    reply_text, sources = extract_bot_reply(last_msg)
    if use_cse:
        sources = collect_cse_sources(response["messages"])

    # 5. 토큰 및 비용 통계 누적
    turn_stats = calculate_token_usage_and_cost(response["messages"], req.model)

    sess = SESSION_STATS.setdefault(
        thread_id,
        {
            "total_input": 0,
            "total_output": 0,
            "total_cost_krw": 0.0,
            "total_cost_usd": 0.0,
        },
    )
    sess["total_input"] += turn_stats["input_tokens"]
    sess["total_output"] += turn_stats["output_tokens"]
    sess["total_cost_krw"] += turn_stats["cost_krw"]
    sess["total_cost_usd"] += turn_stats["cost_usd"]

    return {
        "reply": reply_text,
        "thread_id": thread_id,
        "sources": sources,
        "token_stats": {
            "last_input": turn_stats["input_tokens"],
            "last_output": turn_stats["output_tokens"],
            "total_input": sess["total_input"],
            "total_output": sess["total_output"],
            "total_tokens": sess["total_input"] + sess["total_output"],
            "total_cost_krw": round(sess["total_cost_krw"], 4),
            "total_cost_usd": round(sess["total_cost_usd"], 6),
        },
    }


# ==============================================================================
# [5] 브라우저 웹 UI (HTML/CSS/JS) 제공 엔드포인트
# ==============================================================================
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """웹 브라우저에서 직접 대화할 수 있는 모던 채팅 인터페이스를 반환합니다."""
    html_content = """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>💬 AI 하이브리드 챗봇 (FastAPI)</title>
  <style>
    :root {
      --primary: #2563eb;
      --primary-hover: #1d4ed8;
      --bg-main: #f8fafc;
      --bg-card: #ffffff;
      --border-color: #e2e8f0;
      --text-main: #0f172a;
      --text-muted: #64748b;
      --bot-msg-bg: #f1f5f9;
      --user-msg-bg: #2563eb;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans KR", sans-serif;
      background-color: var(--bg-main);
      color: var(--text-main);
      display: flex;
      height: 100vh;
      overflow: hidden;
    }
    /* 사이드바 */
    aside {
      width: 320px;
      background-color: var(--bg-card);
      border-right: 1px solid var(--border-color);
      display: flex;
      flex-direction: column;
      padding: 24px 20px;
      gap: 20px;
      overflow-y: auto;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 1.2rem;
      font-weight: 700;
      color: var(--primary);
    }
    .section-title {
      font-size: 0.85rem;
      text-transform: uppercase;
      font-weight: 700;
      letter-spacing: 0.5px;
      color: var(--text-muted);
      margin-bottom: 8px;
    }
    .option-group {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .radio-label {
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 0.92rem;
      cursor: pointer;
      padding: 8px 10px;
      border-radius: 8px;
      border: 1px solid var(--border-color);
      transition: background 0.15s;
    }
    .radio-label:hover { background: #f8fafc; }
    .radio-label input:checked + span { font-weight: 600; color: var(--primary); }
    /* 토글 스위치 */
    .toggle-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 10px 12px;
      background: #f8fafc;
      border-radius: 8px;
      border: 1px solid var(--border-color);
    }
    .toggle-switch {
      position: relative;
      display: inline-block;
      width: 44px;
      height: 24px;
    }
    .toggle-switch input { opacity: 0; width: 0; height: 0; }
    .slider {
      position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0;
      background-color: #cbd5e1;
      transition: .3s;
      border-radius: 24px;
    }
    .slider:before {
      position: absolute; content: ""; height: 18px; width: 18px; left: 3px; bottom: 3px;
      background-color: white;
      transition: .3s;
      border-radius: 50%;
    }
    input:checked + .slider { background-color: #10b981; }
    input:checked + .slider:before { transform: translateX(20px); }
    .search-hint {
      font-size: 0.78rem;
      color: var(--text-muted);
      margin-top: 4px;
      line-height: 1.4;
    }
    .info-card {
      background: #f8fafc;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 12px;
      font-size: 0.82rem;
      line-height: 1.6;
    }
    .btn-reset {
      background: #fee2e2;
      color: #dc2626;
      border: 1px solid #fca5a5;
      padding: 10px;
      border-radius: 8px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      transition: background 0.15s;
    }
    .btn-reset:hover { background: #fecaca; }

    /* 메인 채팅 영역 */
    main {
      flex: 1;
      display: flex;
      flex-direction: column;
      height: 100vh;
      position: relative;
    }
    header {
      padding: 16px 24px;
      background: var(--bg-card);
      border-bottom: 1px solid var(--border-color);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .header-badges {
      display: flex;
      gap: 8px;
      font-size: 0.8rem;
    }
    .badge {
      padding: 4px 10px;
      border-radius: 12px;
      background: #e0f2fe;
      color: #0369a1;
      font-weight: 600;
    }
    .badge.search-on {
      background: #d1fae5;
      color: #065f46;
    }
    .chat-box {
      flex: 1;
      overflow-y: auto;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }
    .msg-row {
      display: flex;
      gap: 12px;
      max-width: 80%;
    }
    .msg-row.user {
      align-self: flex-end;
      flex-direction: row-reverse;
    }
    .msg-row.bot {
      align-self: flex-start;
    }
    .avatar {
      width: 36px;
      height: 36px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.1rem;
      flex-shrink: 0;
    }
    .avatar.bot { background: #e2e8f0; }
    .avatar.user { background: #bfdbfe; }
    .bubble {
      padding: 12px 16px;
      border-radius: 14px;
      font-size: 0.95rem;
      line-height: 1.55;
      word-break: break-word;
      white-space: pre-wrap;
    }
    .msg-row.user .bubble {
      background: var(--user-msg-bg);
      color: white;
      border-bottom-right-radius: 2px;
    }
    .msg-row.bot .bubble {
      background: var(--bot-msg-bg);
      color: var(--text-main);
      border: 1px solid var(--border-color);
      border-bottom-left-radius: 2px;
    }
    .sources-box {
      margin-top: 10px;
      padding-top: 8px;
      border-top: 1px dashed #cbd5e1;
      font-size: 0.8rem;
      color: #475569;
    }
    .source-link {
      display: inline-block;
      margin-right: 8px;
      margin-top: 4px;
      color: var(--primary);
      text-decoration: none;
      background: white;
      padding: 3px 8px;
      border-radius: 6px;
      border: 1px solid var(--border-color);
    }
    .source-link:hover { text-decoration: underline; }

    /* 대시보드 하단 고정 영역 */
    footer {
      background: var(--bg-card);
      border-top: 1px solid var(--border-color);
      padding: 14px 24px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .metrics-bar {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
    }
    .metric-card {
      background: #f8fafc;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 8px 12px;
    }
    .metric-label { font-size: 0.72rem; color: var(--text-muted); font-weight: 600; }
    .metric-val { font-size: 1.05rem; font-weight: 700; color: var(--text-main); margin-top: 2px; }

    .input-bar {
      display: flex;
      gap: 10px;
    }
    .input-bar input {
      flex: 1;
      padding: 12px 16px;
      border-radius: 10px;
      border: 1px solid var(--border-color);
      font-size: 0.95rem;
      outline: none;
      transition: border 0.15s;
    }
    .input-bar input:focus { border-color: var(--primary); }
    .btn-send {
      background: var(--primary);
      color: white;
      border: none;
      padding: 0 20px;
      border-radius: 10px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: background 0.15s;
    }
    .btn-send:hover { background: var(--primary-hover); }
    .btn-send:disabled { background: #94a3b8; cursor: not-allowed; }

    /* 로딩 애니메이션 */
    .typing {
      display: inline-flex;
      gap: 4px;
      padding: 6px 10px;
    }
    .typing span {
      width: 6px; height: 6px; background: #64748b; border-radius: 50%;
      animation: bounce 1.2s infinite ease-in-out;
    }
    .typing span:nth-child(2) { animation-delay: 0.2s; }
    .typing span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes bounce {
      0%, 80%, 100% { transform: translateY(0); }
      40% { transform: translateY(-6px); }
    }
  </style>
</head>
<body>

  <!-- 사이드바 -->
  <aside>
    <div class="brand">💬 AI 하이브리드</div>

    <div>
      <div class="section-title">⚙️ 모델 선택</div>
      <div class="option-group">
        <label class="radio-label">
          <input type="radio" name="model" value="gemini" checked onchange="updateModelUI()">
          <span>Google Gemini (Vertex AI ADC)</span>
        </label>
        <label class="radio-label">
          <input type="radio" name="model" value="ollama" onchange="updateModelUI()">
          <span>로컬 Ollama (gemma4:e2b)</span>
        </label>
      </div>
    </div>

    <div>
      <div class="section-title">🌐 구글 검색 연동</div>
      <div class="option-group">
        <label class="radio-label">
          <input type="radio" name="searchMode" value="off" checked onchange="updateModelUI()">
          <span>검색 OFF</span>
        </label>
        <label class="radio-label">
          <input type="radio" name="searchMode" value="grounding" onchange="updateModelUI()">
          <span>Vertex AI Grounding with Google Search</span>
        </label>
        <label class="radio-label">
          <input type="radio" name="searchMode" value="cse" onchange="updateModelUI()">
          <span>Custom Search (google_cse_search)</span>
        </label>
      </div>
      <div class="search-hint" id="searchHint">
        ⚪ <b>검색 OFF</b>: 모델 자체 지식으로 답변합니다.
      </div>
    </div>

    <div>
      <div class="section-title">ℹ️ 시스템 정보</div>
      <div class="info-card">
        <div><b>프로젝트:</b> <span id="infoProject">iceu-songpa04</span></div>
        <div><b>리전:</b> <span id="infoLocation">us-central1</span></div>
        <div><b>세션 ID:</b> <code id="infoThread">-</code></div>
        <div><b>메모리:</b> <span id="infoMsgCount">0개 메시지</span></div>
      </div>
    </div>

    <button class="btn-reset" onclick="resetChat()">
      🗑️ 대화 초기화 (새 세션)
    </button>
  </aside>

  <!-- 메인 채팅창 -->
  <main>
    <header>
      <div>
        <h2 style="font-size:1.15rem; font-weight:700;">💬 AI 하이브리드 챗봇</h2>
        <span style="font-size:0.8rem; color:var(--text-muted);">FastAPI Backend & LangChain InMemorySaver</span>
      </div>
      <div class="header-badges">
        <span class="badge" id="badgeModel">Gemini 2.5 Flash</span>
        <span class="badge" id="badgeSearch">검색 OFF</span>
      </div>
    </header>

    <div class="chat-box" id="chatBox">
      <!-- 봇 환영 메시지 -->
      <div class="msg-row bot">
        <div class="avatar bot">🤖</div>
        <div class="bubble">안녕하세요! 무엇이든 물어보세요. 사이드바에서 구글 실시간 검색과 모델을 변경할 수 있습니다.</div>
      </div>
    </div>

    <!-- 하단 토큰/비용 대시보드 및 입력창 -->
    <footer>
      <div class="metrics-bar">
        <div class="metric-card">
          <div class="metric-label">📥 이번 턴 입력</div>
          <div class="metric-val" id="metricLastInput">0 토큰</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">📤 이번 턴 출력</div>
          <div class="metric-val" id="metricLastOutput">0 토큰</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">📊 세션 누적 토큰</div>
          <div class="metric-val" id="metricTotalTokens">0 토큰</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">💰 누적 예상 비용</div>
          <div class="metric-val" id="metricTotalCost">0.000원</div>
        </div>
      </div>

      <div class="input-bar">
        <input type="text" id="userInput" placeholder="메시지를 입력하세요... (Enter 키로 전송)" onkeydown="if(event.key==='Enter') sendMessage()">
        <button class="btn-send" id="btnSend" onclick="sendMessage()">전송</button>
      </div>
    </footer>
  </main>

  <script>
    let currentThreadId = "";
    let messageCount = 0;

    // 세션 초기화
    async function resetChat() {
      try {
        const res = await fetch("/api/reset", { method: "POST" });
        const data = await res.json();
        currentThreadId = data.thread_id;
        document.getElementById("infoThread").textContent = currentThreadId;
        messageCount = 0;
        document.getElementById("infoMsgCount").textContent = "0개 메시지";

        // 화면 초기화
        const chatBox = document.getElementById("chatBox");
        chatBox.innerHTML = `
          <div class="msg-row bot">
            <div class="avatar bot">🤖</div>
            <div class="bubble">대화가 초기화되었습니다. 새로운 질문을 입력해 주세요!</div>
          </div>
        `;
        document.getElementById("metricLastInput").textContent = "0 토큰";
        document.getElementById("metricLastOutput").textContent = "0 토큰";
        document.getElementById("metricTotalTokens").textContent = "0 토큰";
        document.getElementById("metricTotalCost").textContent = "0.000원";
      } catch (e) {
        alert("세션 초기화 실패: " + e);
      }
    }

    // 모델/검색 UI 상태 업데이트
    function updateModelUI() {
      const model = document.querySelector('input[name="model"]:checked').value;
      const mode = document.querySelector('input[name="searchMode"]:checked').value;

      const badgeModel = document.getElementById("badgeModel");
      const badgeSearch = document.getElementById("badgeSearch");
      const searchHint = document.getElementById("searchHint");

      badgeModel.textContent = model === "gemini" ? "Gemini 2.5 Flash" : "Ollama (gemma4:e2b)";

      if (mode === "grounding") {
        badgeSearch.textContent = "🌐 Grounding";
        badgeSearch.className = "badge search-on";
        searchHint.innerHTML = model === "gemini"
          ? "🟢 <b>Vertex AI Grounding</b>: Gemini가 구글 검색을 직접 수행해 답변합니다."
          : "⚠️ <b>알림</b>: Grounding은 Gemini 전용입니다. Ollama에서는 적용되지 않습니다.";
      } else if (mode === "cse") {
        badgeSearch.textContent = "🔎 Custom Search";
        badgeSearch.className = "badge search-on";
        searchHint.innerHTML = "🟢 <b>Custom Search</b>: google_cse_search 도구로 Programmable Search Engine을 호출합니다. (.env의 GOOGLE_CSE_API_KEY / GOOGLE_CSE_ID 필요)";
      } else {
        badgeSearch.textContent = "검색 OFF";
        badgeSearch.className = "badge";
        searchHint.innerHTML = "⚪ <b>검색 OFF</b>: 모델 자체 학습 지식으로 답변합니다.";
      }
    }

    // 메시지 전송
    async function sendMessage() {
      const input = document.getElementById("userInput");
      const msg = input.value.trim();
      if (!msg) return;

      const model = document.querySelector('input[name="model"]:checked').value;
      const searchMode = document.querySelector('input[name="searchMode"]:checked').value;
      const btn = document.getElementById("btnSend");
      const chatBox = document.getElementById("chatBox");

      // 사용자 메시지 렌더링
      input.value = "";
      btn.disabled = true;

      const userRow = document.createElement("div");
      userRow.className = "msg-row user";
      userRow.innerHTML = `
        <div class="avatar user">👤</div>
        <div class="bubble">${escapeHtml(msg)}</div>
      `;
      chatBox.appendChild(userRow);

      // 로딩 말풍선 표시
      const loadingRow = document.createElement("div");
      loadingRow.className = "msg-row bot";
      loadingRow.id = "loadingMsg";
      loadingRow.innerHTML = `
        <div class="avatar bot">🤖</div>
        <div class="bubble">
          <div class="typing"><span></span><span></span><span></span></div>
        </div>
      `;
      chatBox.appendChild(loadingRow);
      chatBox.scrollTop = chatBox.scrollHeight;

      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: msg,
            model: model,
            search_mode: searchMode,
            thread_id: currentThreadId || null,
          }),
        });

        if (!res.ok) {
          const errData = await res.json();
          throw new Error(errData.detail || "서버 응답 오류");
        }

        const data = await res.json();
        currentThreadId = data.thread_id;
        document.getElementById("infoThread").textContent = currentThreadId;

        // 로딩 제거
        loadingRow.remove();

        // 봇 메시지 렌더링
        let sourcesHtml = "";
        if (data.sources && data.sources.length > 0) {
          sourcesHtml = '<div class="sources-box"><b>🌐 실시간 검색 출처:</b><br>';
          data.sources.forEach((s) => {
            sourcesHtml += `<a class="source-link" href="${s.uri}" target="_blank" rel="noopener">🔗 ${escapeHtml(s.title)}</a>`;
          });
          sourcesHtml += '</div>';
        }

        const botRow = document.createElement("div");
        botRow.className = "msg-row bot";
        botRow.innerHTML = `
          <div class="avatar bot">🤖</div>
          <div class="bubble">
            ${escapeHtml(data.reply)}
            ${sourcesHtml}
          </div>
        `;
        chatBox.appendChild(botRow);
        chatBox.scrollTop = chatBox.scrollHeight;

        // 메트릭 업데이트
        const stats = data.token_stats;
        document.getElementById("metricLastInput").textContent = stats.last_input.toLocaleString() + " 토큰";
        document.getElementById("metricLastOutput").textContent = stats.last_output.toLocaleString() + " 토큰";
        document.getElementById("metricTotalTokens").textContent = stats.total_tokens.toLocaleString() + " 토큰";
        if (model === "ollama") {
          document.getElementById("metricTotalCost").textContent = "0원 (로컬 무료)";
        } else {
          document.getElementById("metricTotalCost").textContent = stats.total_cost_krw.toFixed(3) + "원 ($" + stats.total_cost_usd.toFixed(5) + ")";
        }

        // 메모리 메시지 카운트 갱신
        messageCount += 2;
        document.getElementById("infoMsgCount").textContent = messageCount + "개 메시지";

      } catch (err) {
        loadingRow.remove();
        const errRow = document.createElement("div");
        errRow.className = "msg-row bot";
        errRow.innerHTML = `
          <div class="avatar bot">⚠️</div>
          <div class="bubble" style="background:#fee2e2; color:#b91c1c;">
            오류 발생: ${escapeHtml(err.message)}
          </div>
        `;
        chatBox.appendChild(errRow);
      } finally {
        btn.disabled = false;
        input.focus();
      }
    }

    function escapeHtml(str) {
      if (!str) return "";
      return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    }

    // 초기 실행
    window.addEventListener("DOMContentLoaded", () => {
      resetChat();
      updateModelUI();
    });
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


# ==============================================================================
# [6] 메인 실행 엔트리포인트 (python app.py 바로 실행 가능)
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 FastAPI AI 하이브리드 챗봇 서버를 시작합니다.")
    print("👉 웹 브라우저 접속 주소: http://127.0.0.1:8000")
    print("👉 Swagger API 문서: http://127.0.0.1:8000/docs")
    print("=" * 60)
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
