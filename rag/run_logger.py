"""RAG 실행 로그(JSONL) 기록 모듈

'인덱싱 시작' / '질문하기' 1회마다 설정 조합과 결과(페이지·청크 수, 검색 결과, 토큰·비용, 소요 시간)를
한 줄 JSON으로 남겨, 나중에 최적 조합(OCR/임베딩/캐시/LLM/청크/top_k 등)을 분석할 수 있게 한다.

- rag/logs/index_runs.jsonl : 인덱싱 1회 = 1줄
- rag/logs/query_runs.jsonl : 질문 1회 = 1줄

분석 예: pandas.json_normalize([json.loads(l) for l in open(path, encoding="utf-8")])
"""

import hashlib
import json
import re
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG_DIR = Path(__file__).resolve().parent / "logs"
INDEX_LOG_PATH = LOG_DIR / "index_runs.jsonl"
QUERY_LOG_PATH = LOG_DIR / "query_runs.jsonl"
EVAL_SET_PATH = Path(__file__).resolve().parent / "han_gang_35.json"
SCHEMA_VERSION = 1

_write_lock = threading.Lock()
_eval_index: Optional[Dict[str, Dict[str, Any]]] = None


def log_path(kind: str) -> Path:
    return INDEX_LOG_PATH if kind == "index" else QUERY_LOG_PATH


def read_runs(kind: str, limit: int = 500) -> List[Dict[str, Any]]:
    """로그 파일을 최신순으로 읽기 (깨진 줄은 건너뜀)"""
    path = log_path(kind)
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records[::-1][:limit]


def _avg(values: List[Any]) -> Optional[float]:
    nums = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return round(sum(nums) / len(nums), 4) if nums else None


def summarize_query_runs(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """설정 조합(LLM/top_k/질의 확장/리랭킹/임베딩 캐시/chunk_size)별 평균 비용·시간·품질 집계"""
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in records:
        if r.get("status") != "success":
            continue
        s = r.get("settings") or {}
        key = (
            s.get("llm_model"), s.get("top_k"), s.get("use_query_expansion"), s.get("use_rerank"),
            s.get("use_embedding_cache"), s.get("chunk_size"),
        )
        groups.setdefault(key, []).append(r)

    rows = []
    for (llm, top_k, qe, rr, cache, chunk), items in groups.items():
        evals = [i["eval"] for i in items if i.get("eval")]
        hits = [e["retrieval_page_hit"] for e in evals if e.get("retrieval_page_hit") is not None]
        rows.append(
            {
                "llm_model": llm,
                "top_k": top_k,
                "use_query_expansion": qe,
                "use_rerank": rr,
                "use_embedding_cache": cache,
                "chunk_size": chunk,
                "runs": len(items),
                "eval_runs": len(evals),
                "avg_cost_krw": _avg([(i.get("usage") or {}).get("total_cost_krw") for i in items]),
                "avg_wall_sec": _avg([(i.get("timings_sec") or {}).get("wall") for i in items]),
                "avg_answer_len": _avg([i.get("answer_len") for i in items]),
                "page_hit_rate": round(sum(hits) / len(hits), 3) if hits else None,
                "avg_keyword_recall": _avg([e.get("keyword_recall") for e in evals]),
            }
        )
    rows.sort(key=lambda x: (-(x["avg_keyword_recall"] or -1), x["avg_cost_krw"] or 0))
    return rows


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _append(path: Path, record: Dict[str, Any]) -> None:
    """로그 기록 실패가 서비스 응답을 막지 않도록 예외는 콘솔 경고로만 처리"""
    try:
        LOG_DIR.mkdir(exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _write_lock, open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"  [경고] 실행 로그 기록 실패({path.name}): {e}")


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _file_sha256(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except Exception:
        return None


def _summarize_usage(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """단계별 사용량 목록 -> 합계 및 단계별 집계"""
    by_stage: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        s = by_stage.setdefault(
            e["stage"],
            {"model": e["model"], "calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "estimated": False},
        )
        s["calls"] += 1
        s["input_tokens"] += e.get("input_tokens", 0)
        s["output_tokens"] += e.get("output_tokens", 0)
        s["cost_usd"] = round(s["cost_usd"] + e.get("cost_usd", 0.0), 7)
        s["estimated"] = s["estimated"] or bool(e.get("estimated"))
    return {
        "total_input_tokens": sum(e.get("input_tokens", 0) for e in entries),
        "total_output_tokens": sum(e.get("output_tokens", 0) for e in entries),
        "total_cost_usd": round(sum(e.get("cost_usd", 0.0) for e in entries), 7),
        "total_cost_krw": round(sum(e.get("cost_krw", 0.0) for e in entries), 4),
        "by_stage": by_stage,
    }


def _indexed_files(pipeline: Any) -> List[Dict[str, Any]]:
    try:
        with pipeline.vector_store._get_connection() as conn:
            rows = conn.execute(
                "SELECT file_name, total_pages, total_chunks, indexed_at FROM indexed_files ORDER BY indexed_at;"
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _db_state(pipeline: Any) -> Dict[str, Any]:
    try:
        total_chunks = pipeline.vector_store.count()
    except Exception:
        total_chunks = None
    return {"total_chunks_in_db": total_chunks, "indexed_files": _indexed_files(pipeline)}


# =========================================================================
# 인덱싱 로그
# =========================================================================


def log_index_run(
    *,
    run_id: str,
    pipeline: Any,
    file_path: str,
    params: Dict[str, Any],
    wall_time: float,
    result: Optional[Dict[str, Any]] = None,
    aux_usage: Optional[List[Dict[str, Any]]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = pipeline.config
    result = result or {}
    s1 = result.get("step1_load") or {}
    s2 = result.get("step2_split") or {}
    s3 = result.get("step3_embed") or {}
    s4 = result.get("step4_store") or {}
    ocr_prompt = params.get("ocr_prompt") or cfg.ocr_prompt

    if error:
        status = "error"
    elif result.get("skipped"):
        status = "skipped"
    else:
        status = "success"

    record = {
        "schema_version": SCHEMA_VERSION,
        "event": "index",
        "run_id": run_id,
        "timestamp": _now(),
        "status": status,
        "skip_reason": result.get("skip_reason") or None,
        "error": error,
        "file": {
            "name": Path(file_path).name,
            "sha256": _file_sha256(file_path),
            "size_kb": s1.get("file_size_kb") or round(Path(file_path).stat().st_size / 1024, 1)
            if Path(file_path).exists()
            else None,
        },
        "settings": {
            "ocr_model": cfg.ocr_model,
            "ocr_prompt_is_default": ocr_prompt.strip() == _default_ocr_prompt(pipeline),
            "ocr_prompt_len": len(ocr_prompt),
            "ocr_prompt_hash": _short_hash(ocr_prompt),
            "embedding_model": cfg.embedding_model,
            "use_embedding_cache": cfg.use_embedding_cache,
            "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap,
            "clear_existing": params.get("clear_existing"),
            "detect_subset": params.get("detect_subset"),
            "force_reindex": params.get("force_reindex"),
        },
        "pages": {
            "pdf_total_pages": (result.get("page_plan") or {}).get("total_pages") or s1.get("total_pages"),
            "ocr_pages": len(s1.get("pages_content") or []) if s1 else 0,
            "skipped_pages": (result.get("page_plan") or {}).get("skipped_pages"),
            "total_chars": s1.get("total_chars"),
        },
        "chunks": {
            "chunk_count": s2.get("chunk_count"),
            "min_len": s2.get("min_len"),
            "max_len": s2.get("max_len"),
            "avg_len": s2.get("avg_len"),
        },
        "embedding": {
            "vector_count": s3.get("vector_count"),
            "dimension": s3.get("dimension"),
            "cache_hit_count": s3.get("cache_hit_count"),
            "cache_miss_count": s3.get("cache_miss_count"),
        },
        "store": {
            "saved_count": s4.get("saved_count"),
            "db_file_size_kb": s4.get("file_size_kb"),
        },
        "timings_sec": {
            "a1_ocr": s1.get("elapsed"),
            "a2_split": s2.get("elapsed"),
            "a3_embed": s3.get("elapsed"),
            "a4_store": s4.get("elapsed"),
            "wall": round(wall_time, 2),
        },
        "usage": {**_summarize_usage(aux_usage or []), "calls": aux_usage or []},
        "db_after": _db_state(pipeline),
    }
    _append(INDEX_LOG_PATH, record)
    return record


def _default_ocr_prompt(pipeline: Any) -> str:
    # 패키지/스크립트 실행 방식과 무관하게 파이프라인이 로드된 모듈의 기본 프롬프트를 사용
    module = sys.modules.get(type(pipeline).__module__)
    return (getattr(module, "DEFAULT_OCR_PROMPT", "") or "").strip()


# =========================================================================
# 질문 로그
# =========================================================================


def _load_eval_index() -> Dict[str, Dict[str, Any]]:
    global _eval_index
    if _eval_index is None:
        _eval_index = {}
        try:
            for item in json.loads(EVAL_SET_PATH.read_text(encoding="utf-8")):
                _eval_index[item["question"].strip()] = item
        except Exception:
            pass
    return _eval_index


def _page_range(text: str) -> Optional[List[int]]:
    nums = [int(n) for n in re.findall(r"\d+", text or "")]
    if not nums:
        return None
    return [min(nums), max(nums)]


def _eval_match(question: str, answer: str, retrieved_pages: List[Any]) -> Optional[Dict[str, Any]]:
    """질문이 35문항 평가셋과 일치하면 정답 키워드 재현율과 검색 페이지 적중 여부를 함께 기록"""
    item = _load_eval_index().get(question.strip())
    if not item:
        return None
    keywords = item.get("key_keywords") or []
    hits = [k for k in keywords if k in answer]
    pr = _page_range(item.get("page_range", ""))
    pages = [p for p in retrieved_pages if isinstance(p, int)]
    return {
        "eval_id": item.get("id"),
        "category": item.get("category"),
        "difficulty": item.get("difficulty"),
        "expected_page_range": pr,
        "retrieval_page_hit": (any(pr[0] <= p <= pr[1] for p in pages) if pr else None),
        "keyword_total": len(keywords),
        "keyword_hits": hits,
        "keyword_recall": round(len(hits) / len(keywords), 3) if keywords else None,
    }


def log_query_run(
    *,
    run_id: str,
    pipeline: Any,
    question: str,
    params: Dict[str, Any],
    wall_time: float,
    result: Optional[Dict[str, Any]] = None,
    aux_usage: Optional[List[Dict[str, Any]]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = pipeline.config
    result = result or {}
    s1 = result.get("step1_query") or {}
    s2 = result.get("step2_retrieval") or {}
    s3 = result.get("step3_prompt") or {}
    s4 = result.get("step4_generation") or {}
    answer = result.get("answer") or ""

    usage_entries = list(aux_usage or [])
    gen_usage = s4.get("usage") or {}
    if gen_usage:
        usage_entries.append(
            {
                "stage": "답변 생성",
                "model": s4.get("model"),
                "input_tokens": gen_usage.get("input_tokens", 0),
                "output_tokens": gen_usage.get("output_tokens", 0),
                "cost_usd": gen_usage.get("cost_usd", 0.0),
                "cost_krw": gen_usage.get("cost_krw", 0.0),
                "estimated": False,
            }
        )

    retrieved = [
        {
            "rank": r.get("rank"),
            "source": r.get("source"),
            "page": r.get("page"),
            "chunk_index": r.get("chunk_index"),
            "score": r.get("score"),
            "dense_rank": r.get("dense_rank"),
            "dense_similarity": r.get("dense_similarity"),
            "bm25_rank": r.get("bm25_rank"),
            "bm25_score": r.get("bm25_score"),
            "char_count": len(r.get("full_content") or ""),
        }
        for r in (s2.get("results_preview") or [])
    ]
    for r, (doc, _) in zip(retrieved, s2.get("retrieved_docs") or []):
        r["pre_rerank_rank"] = doc.metadata.get("pre_rerank_rank")

    record = {
        "schema_version": SCHEMA_VERSION,
        "event": "query",
        "run_id": run_id,
        "timestamp": _now(),
        "status": "error" if error else "success",
        "error": error,
        "question": question,
        "settings": {
            "llm_model": params.get("llm_model") or cfg.llm_model,
            "aux_llm_model": cfg.llm_model,  # 질의 확장/리랭킹에 쓰이는 기본 LLM
            "temperature": params.get("temperature") if params.get("temperature") is not None else cfg.temperature,
            "top_k": params.get("top_k") if params.get("top_k") is not None else cfg.top_k,
            "candidate_k": cfg.candidate_k,
            "use_query_expansion": cfg.use_query_expansion,
            "use_rerank": cfg.use_rerank,
            "embedding_model": cfg.embedding_model,
            "use_embedding_cache": cfg.use_embedding_cache,
            "ocr_model": cfg.ocr_model,
            "chunk_size": cfg.chunk_size,  # 마지막 인덱싱 시 설정값 (DB 내 청크가 다른 설정으로 만들어졌을 수 있음)
            "chunk_overlap": cfg.chunk_overlap,
        },
        "retrieval": {
            "expanded_keywords": s1.get("expanded_keywords"),
            "query_embedding_cache_hit": s1.get("cache_hit"),
            "retrieved_count": s2.get("retrieved_count"),
            "retrieved": retrieved,
            "context_chars": len(s3.get("context_str") or ""),
        },
        "answer": answer,
        "answer_len": len(answer),
        "eval": _eval_match(question, answer, [r["page"] for r in retrieved]) if answer else None,
        "timings_sec": {
            "b0_expand": s1.get("expand_elapsed"),
            "b1_embed": s1.get("elapsed"),
            "b2_search_rerank": s2.get("elapsed"),
            "b3_prompt": s3.get("elapsed"),
            "b4_generate": s4.get("latency"),
            "wall": round(wall_time, 2),
        },
        "usage": {**_summarize_usage(usage_entries), "calls": usage_entries},
        "db_state": _db_state(pipeline),
    }
    _append(QUERY_LOG_PATH, record)
    return record
