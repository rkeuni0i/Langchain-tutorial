"""FastAPI 기반 LangChain RAG 웹 애플리케이션

첨부 다이어그램의 단계별(A1~A4, B1~B4) 파이프라인 출력을
웹 대시보드와 REST API(/docs)를 통해 제공합니다.
"""

import json
import os
import queue
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional

# 현재 디렉토리 sys.path 보정
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Windows 콘솔 한글/특수문자 인코딩 보정
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

try:
    from .rag_pipeline import DEFAULT_OCR_PROMPT, PDFRAGPipeline, RAGConfig
    from .sqlite_vector_store import SQLiteVectorStore
    from .gemini_error_debugger import diagnose_gemini_error, print_gemini_429_diagnosis
except ImportError:
    from rag_pipeline import DEFAULT_OCR_PROMPT, PDFRAGPipeline, RAGConfig
    from sqlite_vector_store import SQLiteVectorStore
    from gemini_error_debugger import diagnose_gemini_error, print_gemini_429_diagnosis

# 업로드 디렉토리 준비 (rag/uploads)
UPLOAD_DIR = CURRENT_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# 기본 전역 파이프라인 인스턴스 (초기 config)
global_config = RAGConfig(
    chunk_size=500,
    chunk_overlap=100,
    top_k=3,
    db_path=str(CURRENT_DIR / "rag_sqlite.db"),
    embedding_model="gemini-embedding-2",
    llm_model="gemini-3.8-flash",
    temperature=0.2,
)
pipeline = PDFRAGPipeline(config=global_config)

app = FastAPI(
    title="LangChain PDF RAG 웹 서비스",
    description="Gemini 임베딩(gemini-embedding-2) 및 LLM(gemini-3.8-flash)과 SQLite 벡터 DB 기반 단계별 RAG 시스템",
    version="1.0.0",
)

# CORS 미들웨어
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    temperature: Optional[float] = None
    llm_model: Optional[str] = None


@app.get("/api/status")
async def get_status():
    """현재 SQLite 벡터 DB 상태 및 파이프라인 설정 조회"""
    total_chunks = pipeline.vector_store.count()
    db_size_kb = (
        os.path.getsize(pipeline.config.db_path) / 1024
        if os.path.exists(pipeline.config.db_path)
        else 0
    )
    return {
        "status": "ready",
        "total_chunks_in_db": total_chunks,
        "db_size_kb": round(db_size_kb, 2),
        "config": {
            "chunk_size": pipeline.config.chunk_size,
            "chunk_overlap": pipeline.config.chunk_overlap,
            "top_k": pipeline.config.top_k,
            "embedding_model": pipeline.config.embedding_model,
            "llm_model": pipeline.config.llm_model,
            "available_llm_models": ["gemini-3.8-flash", "gemma4:e2b"],
            "db_path": pipeline.config.db_path,
            "ocr_prompt": pipeline.config.ocr_prompt,
        },
        "default_ocr_prompt": DEFAULT_OCR_PROMPT,
    }


@app.delete("/api/clear")
async def clear_database():
    """SQLite 벡터 DB 및 문서 캐시 초기화"""
    pipeline.vector_store.clear()
    pipeline.clear_cache()
    return {"message": "SQLite 벡터 데이터베이스와 문서 캐시가 초기화되었습니다.", "total_chunks": 0}


@app.get("/api/document-pages")
async def get_document_pages():
    """현재 인덱싱된 문서의 페이지별 전체 내용 및 메타데이터 조회"""
    return pipeline.get_document_pages()


def _format_index_response(
    result: Dict[str, Any],
    file_name: str,
    chunk_size: int,
    chunk_overlap: int,
    pipeline: Any,
) -> Dict[str, Any]:
    """인덱싱 완료/스킵 결과를 프론트엔드 반환 규격으로 포맷팅 (임베딩 캐시 통계 포함)"""
    is_skipped = result.get("skipped", False)
    payload = {
        "success": True,
        "skipped": is_skipped,
        "skipped_as_subset": result.get("skipped_as_subset", False),
        "skipped_as_duplicate": result.get("skipped_as_duplicate", False),
        "skip_reason": result.get("skip_reason", ""),
        "message": result.get("message", ""),
        "subset_info": result.get("subset_info"),
        "file_name": file_name,
        "config": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_model": pipeline.config.embedding_model,
        },
    }

    s1 = result.get("step1_load")
    if s1:
        payload["step_a1_load"] = {
            "file_name": s1.get("file_name", file_name),
            "file_size_kb": s1.get("file_size_kb", 0),
            "total_pages": s1.get("total_pages", 0),
            "total_chars": s1.get("total_chars", 0),
            "sample_preview": s1.get("sample_preview", ""),
            "pages_content": s1.get("pages_content", []),
            "elapsed": s1.get("elapsed", 0),
        }
    else:
        payload["step_a1_load"] = None

    s2 = result.get("step2_split")
    if s2:
        payload["step_a2_split"] = {
            "chunk_count": s2.get("chunk_count", 0),
            "min_len": s2.get("min_len", 0),
            "max_len": s2.get("max_len", 0),
            "avg_len": s2.get("avg_len", 0),
            "sample_first": s2.get("sample_first", ""),
            "sample_last": s2.get("sample_last", ""),
            "elapsed": s2.get("elapsed", 0),
            "chunks": s2.get("chunks") or s2.get("chunk_details") or [],
        }
    else:
        payload["step_a2_split"] = None

    s3 = result.get("step3_embed")
    if s3:
        payload["step_a3_embed"] = {
            "vector_count": s3.get("vector_count", 0),
            "dimension": s3.get("dimension", 0),
            "sample_vector": s3.get("sample_vector", []),
            "elapsed": s3.get("elapsed", 0),
            "cache_hit_count": s3.get("cache_hit_count", 0),
            "cache_miss_count": s3.get("cache_miss_count", 0),
        }
    else:
        payload["step_a3_embed"] = None

    s4 = result.get("step4_store")
    if s4:
        payload["step_a4_store"] = {
            "saved_count": s4.get("saved_count", 0),
            "total_in_db": s4.get("total_in_db", 0),
            "db_path": s4.get("db_path", ""),
            "file_size_kb": s4.get("file_size_kb", 0),
            "elapsed": s4.get("elapsed", 0),
        }
    else:
        payload["step_a4_store"] = None

    return payload


@app.post("/api/index")
async def index_pdf_file(
    file: UploadFile = File(...),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(100),
    clear_existing: bool = Form(False),
    detect_subset: bool = Form(True),
    force_reindex: bool = Form(False),
    ocr_prompt: Optional[str] = Form(None),
):
    """[A단계] PDF 파일 업로드 및 인덱싱 (스텝 A1 ~ A4 실행, 중복 및 하위 문서 자동 필터링 & 임베딩 캐시 지원)"""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드할 수 있습니다.")

    pipeline.config.chunk_size = chunk_size
    pipeline.config.chunk_overlap = chunk_overlap
    if ocr_prompt and ocr_prompt.strip():
        pipeline.config.ocr_prompt = ocr_prompt.strip()

    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = pipeline.index_pdf(
            str(save_path),
            clear_existing=clear_existing,
            ocr_prompt=ocr_prompt,
            detect_subset=detect_subset,
            force_reindex=force_reindex,
        )
        return _format_index_response(result, file.filename, chunk_size, chunk_overlap, pipeline)
    except Exception as e:
        diag = diagnose_gemini_error(e)
        if diag.get("is_429"):
            print_gemini_429_diagnosis(e)
            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "is_429": True,
                    "detail": diag["category_name"],
                    "diagnosis": diag,
                },
            )
        raise HTTPException(status_code=500, detail=f"인덱싱 처리 중 오류 발생: {str(e)}")


@app.post("/api/index-stream")
async def index_pdf_file_stream(
    file: UploadFile = File(...),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(100),
    clear_existing: bool = Form(False),
    detect_subset: bool = Form(True),
    force_reindex: bool = Form(False),
    ocr_prompt: Optional[str] = Form(None),
):
    """[A단계] PDF 파일 업로드 및 실시간 진행률(SSE) 스트리밍 인덱싱 (중복/하위 문서 감지 & 임베딩 캐시)"""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드할 수 있습니다.")

    pipeline.config.chunk_size = chunk_size
    pipeline.config.chunk_overlap = chunk_overlap
    if ocr_prompt and ocr_prompt.strip():
        pipeline.config.ocr_prompt = ocr_prompt.strip()

    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    def event_stream():
        event_queue = queue.Queue()

        def progress_cb(pct: int, step: str, msg: str, data: Optional[Dict[str, Any]] = None):
            event_queue.put({
                "type": "progress",
                "percent": pct,
                "step": step,
                "message": msg,
                "data": data,
            })

        def run_indexing():
            try:
                res = pipeline.index_pdf(
                    str(save_path),
                    clear_existing=clear_existing,
                    ocr_prompt=ocr_prompt,
                    progress_callback=progress_cb,
                    detect_subset=detect_subset,
                    force_reindex=force_reindex,
                )
                final_payload = _format_index_response(
                    res, file.filename, chunk_size, chunk_overlap, pipeline
                )
                event_queue.put({"type": "complete", "result": final_payload})
            except Exception as ex:
                diag = diagnose_gemini_error(ex)
                if diag.get("is_429"):
                    print_gemini_429_diagnosis(ex)
                event_queue.put({
                    "type": "error",
                    "error": str(ex),
                    "diagnosis": diag if diag.get("is_429") else None,
                })
            finally:
                event_queue.put(None)  # 스트림 종료 플래그

        worker_thread = threading.Thread(target=run_indexing, daemon=True)
        worker_thread.start()

        while True:
            try:
                item = event_queue.get(timeout=25)
                if item is None:
                    break
                payload = json.dumps(item, ensure_ascii=False)
                yield f"data: {payload}\n\n"
            except queue.Empty:
                # HTTP 연결 유지용 SSE 주석 전송
                yield ": keep-alive\n\n"
                payload = json.dumps(item, ensure_ascii=False)
                yield f"data: {payload}\n\n"
            except queue.Empty:
                # HTTP 연결 유지용 SSE 주석 전송
                yield ": keep-alive\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/ask")
async def ask_question(request: AskRequest):
    """[B단계] 질문 질의응답 (스텝 B1 ~ B4 실행)"""
    if pipeline.vector_store.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="벡터 DB가 비어 있습니다. 먼저 [A단계]에서 PDF 문서를 인덱싱해 주세요.",
        )

    if request.top_k is not None:
        pipeline.config.top_k = request.top_k
    if request.temperature is not None:
        pipeline.config.temperature = request.temperature
        pipeline.llm.temperature = request.temperature

    try:
        # B단계 런타임 질의응답 파이프라인 실행
        result = pipeline.ask(
            question=request.query,
            top_k=request.top_k,
            temperature=request.temperature,
            llm_model=request.llm_model,
        )

        return {
            "success": True,
            "query": request.query,
            "answer": result["answer"],
            "step_b1_query": {
                "question": result["step1_query"]["question"],
                "dimension": result["step1_query"]["dimension"],
                "sample_vector": result["step1_query"]["sample_vector"],
                "elapsed": result["step1_query"]["elapsed"],
            },
            "step_b2_retrieval": {
                "retrieved_count": result["step2_retrieval"]["retrieved_count"],
                "results_preview": result["step2_retrieval"]["results_preview"],
                "elapsed": result["step2_retrieval"]["elapsed"],
            },
            "step_b3_prompt": {
                "system_instruction_len": len(result["step3_prompt"]["system_instruction"]),
                "context_str_len": len(result["step3_prompt"]["context_str"]),
                "full_prompt_preview": result["step3_prompt"]["full_prompt_text"][:400],
                "elapsed": result["step3_prompt"]["elapsed"],
            },
            "step_b4_generation": {
                "model": result["step4_generation"]["model"],
                "latency": result["step4_generation"]["latency"],
                "answer_len": len(result["step4_generation"]["answer"]),
            },
        }
    except Exception as e:
        diag = diagnose_gemini_error(e)
        if diag.get("is_429"):
            print_gemini_429_diagnosis(e)
            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "is_429": True,
                    "detail": diag["category_name"],
                    "diagnosis": diag,
                },
            )
        raise HTTPException(status_code=500, detail=f"답변 생성 중 오류 발생: {str(e)}")


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """RAG 단계별 시각화 대시보드 웹 UI"""
    html_content = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LangChain PDF RAG 단계별 시각화 시스템</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        .step-card {
            transition: all 0.2s ease-in-out;
        }
        .step-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }
        pre code {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }
    </style>
</head>
<body class="bg-slate-50 text-slate-800 min-h-screen">

    <!-- 헤더 -->
    <header class="bg-white border-b border-slate-200 sticky top-0 z-30 shadow-sm">
        <div class="max-w-7xl mx-auto px-4 py-3 flex flex-wrap justify-between items-center gap-4">
            <div class="flex items-center space-x-3">
                <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-500 to-teal-400 flex items-center justify-center text-white shadow-md">
                    <i class="fa-solid fa-layer-group text-lg"></i>
                </div>
                <div>
                    <h1 class="text-lg font-bold text-slate-900 flex items-center gap-2">
                        LangChain PDF RAG 파이프라인
                        <span class="text-xs bg-emerald-100 text-emerald-800 px-2 py-0.5 rounded-full font-medium">FastAPI Web</span>
                    </h1>
                    <p class="text-xs text-slate-500">Gemini 임베딩-2 & Gemini 3.8 Flash / Ollama Gemma 4:e2b & SQLite 벡터 DB 기반</p>
                </div>
            </div>
            <div class="flex items-center gap-3 text-xs">
                <span id="dbBadge" class="bg-slate-100 border border-slate-200 text-slate-700 px-3 py-1.5 rounded-lg font-semibold flex items-center gap-1.5">
                    <i class="fa-solid fa-database text-emerald-600"></i> 저장된 청크: <span id="chunkCount">0</span>개
                </span>
                <button onclick="clearDb()" class="text-rose-600 hover:bg-rose-50 border border-rose-200 px-2.5 py-1.5 rounded-lg transition" title="DB 초기화">
                    <i class="fa-solid fa-trash-can mr-1"></i> DB 비우기
                </button>
                <a href="/docs" target="_blank" class="text-slate-600 hover:bg-slate-100 border border-slate-200 px-2.5 py-1.5 rounded-lg transition" title="Swagger API Docs">
                    <i class="fa-solid fa-code mr-1"></i> API 명세
                </a>
            </div>
        </div>
    </header>

    <main class="max-w-7xl mx-auto px-4 py-6 space-y-8">

        <!-- ===================================================================== -->
        <!-- 섹션 A. 미리 준비하기 (인덱싱) -->
        <!-- ===================================================================== -->
        <section class="bg-white border border-emerald-200 rounded-2xl shadow-sm p-6 overflow-hidden">
            <div class="flex flex-wrap items-center justify-between pb-4 mb-4 border-b border-emerald-100 gap-3">
                <div class="flex items-center gap-2">
                    <span class="px-2.5 py-1 bg-emerald-600 text-white rounded-md text-xs font-bold tracking-wide">A</span>
                    <h2 class="text-base font-bold text-emerald-950">미리 준비하기 (자료를 등록하거나 변경할 때 · 인덱싱)</h2>
                </div>
                <div class="text-xs text-emerald-700 font-medium">
                    <i class="fa-solid fa-circle-info mr-1"></i> PDF 업로드 시 A1 → A2 → A3 → A4 스텝이 순차 실행됩니다.
                </div>
            </div>

            <!-- 인덱싱 제어 폼 -->
            <form id="indexForm" onsubmit="handleIndex(event); return false;" class="space-y-3.5 mb-6 bg-emerald-50/60 p-4 rounded-xl border border-emerald-100">
                <div class="grid grid-cols-1 md:grid-cols-12 gap-4">
                    <div class="md:col-span-5">
                        <label class="block text-xs font-semibold text-slate-700 mb-1">
                            <i class="fa-solid fa-file-pdf text-red-500 mr-1"></i> PDF 문서 선택
                        </label>
                        <input type="file" id="pdfFileInput" accept=".pdf" required 
                               class="w-full text-xs text-slate-500 file:mr-3 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-emerald-600 file:text-white hover:file:bg-emerald-700 cursor-pointer bg-white border border-slate-200 rounded-lg p-1"/>
                    </div>
                    <div class="md:col-span-3">
                        <label class="block text-xs font-semibold text-slate-700 mb-1">
                            청크 크기 (chunk_size)
                        </label>
                        <div class="flex items-center gap-2">
                            <input type="number" id="chunkSizeInput" value="500" min="100" max="4000" step="50"
                                   class="w-full px-3 py-1.5 text-xs bg-white border border-slate-200 rounded-lg focus:outline-none focus:ring-1 focus:ring-emerald-500"/>
                            <span class="text-xs text-slate-400">자</span>
                        </div>
                    </div>
                    <div class="md:col-span-2">
                        <label class="block text-xs font-semibold text-slate-700 mb-1">
                            청크 오버랩 (overlap)
                        </label>
                        <div class="flex items-center gap-2">
                            <input type="number" id="chunkOverlapInput" value="100" min="0" max="1000" step="10"
                                   class="w-full px-3 py-1.5 text-xs bg-white border border-slate-200 rounded-lg focus:outline-none focus:ring-1 focus:ring-emerald-500"/>
                            <span class="text-xs text-slate-400">자</span>
                        </div>
                    </div>
                    <div class="md:col-span-2 flex items-end">
                        <button type="button" onclick="handleIndex(event)" id="btnIndex"
                                class="w-full bg-emerald-600 hover:bg-emerald-700 text-white font-medium text-xs py-2.5 px-4 rounded-lg shadow-sm transition flex items-center justify-center gap-1.5 cursor-pointer">
                            <i class="fa-solid fa-play"></i> 인덱싱 시작
                        </button>
                    </div>
                </div>

                <!-- [스텝 A1 전용] Gemini OCR 인식 프롬프트 직접 수정 영역 -->
                <div class="border-t border-emerald-200/80 pt-3">
                    <div class="flex items-center justify-between mb-1.5">
                        <label for="ocrPromptInput" class="text-xs font-bold text-emerald-950 flex items-center gap-1.5 cursor-pointer select-none" onclick="toggleOcrPromptArea()">
                            <span class="w-5 h-5 rounded-md bg-emerald-600 text-white flex items-center justify-center text-[10px] shadow-xs">
                                <i class="fa-solid fa-wand-magic-sparkles"></i>
                            </span>
                            <span>[스텝 A1] Gemini OCR 인식 프롬프트 직접 수정</span>
                            <span class="text-[10px] bg-emerald-100 text-emerald-800 px-2 py-0.5 rounded font-normal">사용자 정의 가능</span>
                            <i id="ocrPromptToggleIcon" class="fa-solid fa-chevron-down text-[10px] text-emerald-700 ml-1 transition-transform"></i>
                        </label>
                        <div class="flex items-center gap-2">
                            <span id="ocrPromptCharCount" class="text-[11px] text-slate-400 font-mono">0자</span>
                            <button type="button" onclick="resetOcrPrompt()" 
                                    class="text-[11px] text-slate-600 hover:text-emerald-700 bg-white hover:bg-emerald-50 border border-slate-200 rounded-lg px-2.5 py-1 transition flex items-center gap-1 shadow-2xs cursor-pointer font-medium"
                                    title="권장 기본 프롬프트로 초기화">
                                <i class="fa-solid fa-rotate-left text-[10px] text-emerald-600"></i> 기본값 복원
                            </button>
                        </div>
                    </div>
                    <div id="ocrPromptContainer" class="space-y-1.5">
                        <textarea id="ocrPromptInput" rows="5" oninput="updateOcrPromptCharCount()"
                                  class="w-full p-3 text-xs font-mono bg-white border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-emerald-500/40 focus:border-emerald-500 text-slate-800 leading-relaxed shadow-inner"
                                  placeholder="Gemini OCR 추출 시 모델에 전달할 프롬프트를 입력하세요..
                                  "></textarea>
                        <div class="flex flex-wrap items-center justify-between text-[11px] text-slate-500 px-1 gap-2">
                            <span class="text-emerald-700 font-medium flex items-center gap-1">
                                <i class="fa-solid fa-lightbulb text-amber-500"></i>
                                페이지별 자동 분할 인식을 위해 프롬프트 내에 <b>'--- [PAGE N] ---'</b> 구분자 규칙을 유지해 주세요.
                            </span>
                            <span class="text-slate-400">인덱싱 시작 시 위 프롬프트가 Gemini 모델에 실시간 전달됩니다.</span>
                        </div>
                    </div>
                </div>
            </form>

            <!-- [실시간 SSE 진행률] 인덱싱 프로그레스 바 & 단계별 라이브 피드 -->
            <div id="indexingProgressPanel" class="hidden mb-5 bg-gradient-to-br from-slate-900 via-emerald-950 to-slate-900 text-white rounded-2xl p-5 border border-emerald-500/40 shadow-xl transition-all duration-300">
                <!-- 상단: 현재 스텝 배지 / 상태 메시지 / 퍼센트 & 경과 타이머 -->
                <div class="flex flex-wrap items-center justify-between gap-3 mb-3.5">
                    <div class="flex items-center gap-3">
                        <span id="progressStepBadge" class="px-2.5 py-1 rounded-lg bg-emerald-500 text-slate-950 font-bold text-xs tracking-wider uppercase shadow-xs">
                            스텝 A1
                        </span>
                        <div>
                            <h4 id="progressStatusTitle" class="text-sm font-bold text-slate-100 flex items-center gap-2">
                                <i id="progressStatusIcon" class="fa-solid fa-circle-notch fa-spin text-emerald-400"></i>
                                <span id="progressStatusText">Gemini 3.8 Flash OCR 분석 진행 중...</span>
                            </h4>
                            <p id="progressDetailMessage" class="text-xs text-emerald-200/80 mt-0.5 font-medium">
                                대용량 문서의 이미지 속 시각 요소를 제외하고 순수 텍스트를 추출하고 있습니다.
                            </p>
                        </div>
                    </div>
                    <div class="flex items-center gap-3 text-right">
                        <div class="bg-white/10 px-3 py-1.5 rounded-xl border border-white/10">
                            <span class="text-[10px] text-slate-300 block">경과 시간</span>
                            <span id="progressElapsedTime" class="font-mono text-xs font-bold text-emerald-300">⏱ 00:00</span>
                        </div>
                        <div class="min-w-[70px] bg-emerald-500/10 px-3 py-1 rounded-xl border border-emerald-400/20">
                            <span class="text-[10px] text-emerald-300/80 block">전체 진행률</span>
                            <span id="progressPercentText" class="font-mono text-xl font-black text-emerald-400">0%</span>
                        </div>
                    </div>
                </div>

                <!-- 메인 프로그레스 바 -->
                <div class="w-full bg-slate-800/90 rounded-full h-3.5 p-0.5 border border-white/15 overflow-hidden shadow-inner mb-3.5 relative">
                    <div id="progressBarFill" 
                         class="bg-gradient-to-r from-emerald-500 via-teal-400 to-emerald-300 h-full rounded-full transition-all duration-300 ease-out shadow-sm"
                         style="width: 0%;">
                    </div>
                </div>

                <!-- 4단계 진행 스텝 인디케이터 (미니 플로우) -->
                <div class="grid grid-cols-2 md:grid-cols-4 gap-2 pt-2 border-t border-white/10 text-xs">
                    <div id="stepIndicatorA1" class="flex items-center justify-center gap-1.5 py-1.5 px-2 rounded-xl bg-white/5 border border-white/10 text-slate-300 transition-all">
                        <i class="fa-solid fa-file-lines text-xs"></i>
                        <span class="font-semibold">A1. 문서 읽기</span>
                    </div>
                    <div id="stepIndicatorA2" class="flex items-center justify-center gap-1.5 py-1.5 px-2 rounded-xl bg-white/5 border border-white/10 text-slate-300 transition-all">
                        <i class="fa-solid fa-scissors text-xs"></i>
                        <span class="font-semibold">A2. 청크 분할</span>
                    </div>
                    <div id="stepIndicatorA3" class="flex items-center justify-center gap-1.5 py-1.5 px-2 rounded-xl bg-white/5 border border-white/10 text-slate-300 transition-all">
                        <i class="fa-solid fa-network-wired text-xs"></i>
                        <span class="font-semibold">A3. 임베딩</span>
                    </div>
                    <div id="stepIndicatorA4" class="flex items-center justify-center gap-1.5 py-1.5 px-2 rounded-xl bg-white/5 border border-white/10 text-slate-300 transition-all">
                        <i class="fa-solid fa-database text-xs"></i>
                        <span class="font-semibold">A4. DB 저장</span>
                    </div>
                </div>
            </div>

            <!-- A단계 4개 스텝 카드 그리드 -->
            <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                
                <!-- 스텝 A1 -->
                <div class="step-card bg-slate-50 border border-slate-200 rounded-xl p-4 flex flex-col justify-between">
                    <div>
                        <div class="flex items-center justify-between mb-2">
                            <span class="text-xs font-bold text-emerald-700">스텝 A1</span>
                            <span id="badgeA1" class="text-[10px] px-2 py-0.5 rounded bg-slate-200 text-slate-600 font-medium">대기</span>
                        </div>
                        <h3 class="text-sm font-bold text-slate-800 mb-1">문서 읽기</h3>
                        <p class="text-[11px] text-slate-500 mb-3">Gemini 3.8 Flash OCR (텍스트 추출)</p>
                        <div id="outputA1" class="text-[11px] text-slate-600 space-y-1 bg-white p-2.5 rounded-lg border border-slate-100 min-h-[90px]">
                            <div class="text-slate-400 text-center py-5">PDF 문서를 업로드하세요</div>
                        </div>
                        <!-- 페이지 전체 보기 및 이동 버튼 영역 -->
                        <div id="pageNavMiniA1" class="hidden mt-2 pt-2 border-t border-slate-200 flex-col gap-1.5">
                            <div class="flex items-center justify-between gap-1.5">
                                <button type="button" onclick="openPageModal()" 
                                        class="flex-1 text-[11px] bg-emerald-600 hover:bg-emerald-700 text-white font-medium py-1.5 px-2 rounded-lg shadow-sm transition flex items-center justify-center gap-1.5 cursor-pointer">
                                    <i class="fa-solid fa-expand text-[10px]"></i> 전체 크게 보기
                                </button>
                                <div class="flex items-center gap-1 text-[11px] text-slate-700 font-semibold bg-white border border-slate-200 rounded-lg px-1.5 py-0.5 shadow-xs">
                                    <button type="button" onclick="navigateCardPage(-1)" class="px-1.5 py-0.5 hover:bg-slate-100 rounded text-slate-600 cursor-pointer" title="이전 페이지">◀</button>
                                    <span id="cardPageIndicator" class="px-1 text-[10.5px]">1/1p</span>
                                    <button type="button" onclick="navigateCardPage(1)" class="px-1.5 py-0.5 hover:bg-slate-100 rounded text-slate-600 cursor-pointer" title="다음 페이지">▶</button>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div id="timeA1" class="text-[10px] text-slate-400 text-right mt-2">-</div>
                </div>

                <!-- 스텝 A2 -->
                <div class="step-card bg-slate-50 border border-slate-200 rounded-xl p-4 flex flex-col justify-between">
                    <div>
                        <div class="flex items-center justify-between mb-2">
                            <span class="text-xs font-bold text-emerald-700">스텝 A2</span>
                            <span id="badgeA2" class="text-[10px] px-2 py-0.5 rounded bg-slate-200 text-slate-600 font-medium">대기</span>
                        </div>
                        <h3 class="text-sm font-bold text-slate-800 mb-1">청크로 나누기</h3>
                        <p class="text-[11px] text-slate-500 mb-3">작은 원문 조각 분할</p>
                        <div id="outputA2" class="text-[11px] text-slate-600 space-y-1 bg-white p-2.5 rounded-lg border border-slate-100 min-h-[90px]">
                            <div class="text-slate-400 text-center py-5">청크 분할 결과 대기</div>
                        </div>
                        <!-- 분할 청크 전체 보기 버튼 -->
                        <div id="chunkNavMiniA2" class="hidden mt-2 pt-2 border-t border-slate-200">
                            <button type="button" onclick="openChunksModal()" 
                                    class="w-full text-[11px] bg-emerald-700 hover:bg-emerald-800 text-white font-medium py-1.5 px-2 rounded-lg shadow-sm transition flex items-center justify-center gap-1.5 cursor-pointer">
                                <i class="fa-solid fa-list-check text-[10px]"></i> 분할 청크 전체 문단 보기
                            </button>
                        </div>
                    </div>
                    <div id="timeA2" class="text-[10px] text-slate-400 text-right mt-2">-</div>
                </div>

                <!-- 스텝 A3 -->
                <div class="step-card bg-slate-50 border border-slate-200 rounded-xl p-4 flex flex-col justify-between">
                    <div>
                        <div class="flex items-center justify-between mb-2">
                            <span class="text-xs font-bold text-emerald-700">스텝 A3</span>
                            <span id="badgeA3" class="text-[10px] px-2 py-0.5 rounded bg-slate-200 text-slate-600 font-medium">대기</span>
                        </div>
                        <h3 class="text-sm font-bold text-slate-800 mb-1">임베딩하기</h3>
                        <p class="text-[11px] text-slate-500 mb-3">청크 → 숫자 배열</p>
                        <div id="outputA3" class="text-[11px] text-slate-600 space-y-1 bg-white p-2.5 rounded-lg border border-slate-100 min-h-[90px]">
                            <div class="text-slate-400 text-center py-5">Gemini 임베딩 대기</div>
                        </div>
                    </div>
                    <div id="timeA3" class="text-[10px] text-slate-400 text-right mt-2">-</div>
                </div>

                <!-- 스텝 A4 -->
                <div class="step-card bg-slate-50 border border-slate-200 rounded-xl p-4 flex flex-col justify-between">
                    <div>
                        <div class="flex items-center justify-between mb-2">
                            <span class="text-xs font-bold text-emerald-700">스텝 A4</span>
                            <span id="badgeA4" class="text-[10px] px-2 py-0.5 rounded bg-slate-200 text-slate-600 font-medium">대기</span>
                        </div>
                        <h3 class="text-sm font-bold text-slate-800 mb-1">벡터 스토어 저장</h3>
                        <p class="text-[11px] text-slate-500 mb-3">SQLite (벡터 + 원문 + 출처)</p>
                        <div id="outputA4" class="text-[11px] text-slate-600 space-y-1 bg-white p-2.5 rounded-lg border border-slate-100 min-h-[90px]">
                            <div class="text-slate-400 text-center py-5">SQLite DB 저장 대기</div>
                        </div>
                    </div>
                    <div id="timeA4" class="text-[10px] text-slate-400 text-right mt-2">-</div>
                </div>

            </div>
        </section>


        <!-- ===================================================================== -->
        <!-- 섹션 B. 질문에 답하기 (런타임) -->
        <!-- ===================================================================== -->
        <section class="bg-white border border-amber-200 rounded-2xl shadow-sm p-6 overflow-hidden">
            <div class="flex flex-wrap items-center justify-between pb-4 mb-4 border-b border-amber-100 gap-3">
                <div class="flex items-center gap-2">
                    <span class="px-2.5 py-1 bg-amber-600 text-white rounded-md text-xs font-bold tracking-wide">B</span>
                    <h2 class="text-base font-bold text-amber-950">질문에 답하기 (질문이 들어올 때마다 · 런타임)</h2>
                </div>
                <div class="text-xs text-amber-700 font-medium">
                    <i class="fa-solid fa-circle-info mr-1"></i> 질문 입력 시 B1 → B2 → B3 → B4 스텝이 순차 실행됩니다.
                </div>
            </div>

            <!-- 질문 입력창 폼 -->
            <form id="askForm" onsubmit="handleAsk(event)" class="space-y-3 mb-6 bg-amber-50/60 p-4 rounded-xl border border-amber-100">
                <div class="flex flex-wrap gap-4 items-center">
                    <div class="flex-1 min-w-[280px]">
                        <label class="block text-xs font-semibold text-slate-700 mb-1">
                            <i class="fa-solid fa-comment-dots text-amber-600 mr-1"></i> 질문할 내용을 입력하세요
                        </label>
                        <input type="text" id="queryInput" required
                               placeholder="예: RAG의 주요 이점과 청크 오버랩을 두는 이유는 무엇인가요?"
                               class="w-full px-3.5 py-2 text-xs bg-white border border-slate-200 rounded-lg focus:outline-none focus:ring-1 focus:ring-amber-500 shadow-sm"/>
                    </div>
                    <div class="w-56">
                        <label class="block text-xs font-semibold text-slate-700 mb-1">
                            <i class="fa-solid fa-microchip text-indigo-600 mr-1"></i> LLM 모델 선택
                        </label>
                        <select id="modelSelect" class="w-full px-3 py-2 text-xs bg-white border border-slate-200 rounded-lg focus:outline-none focus:ring-1 focus:ring-amber-500 shadow-sm font-semibold text-slate-700">
                            <option value="gemini-3.8-flash" selected>💎 Gemini 3.8 Flash (Cloud)</option>
                            <option value="gemma4:e2b">🦙 Gemma 4:e2b (Ollama Local)</option>
                        </select>
                    </div>
                    <div class="w-28">
                        <label class="block text-xs font-semibold text-slate-700 mb-1">
                            검색 청크 수
                        </label>
                        <input type="number" id="topKInput" value="4" min="1" max="10"
                               class="w-full px-3 py-2 text-xs bg-white border border-slate-200 rounded-lg focus:outline-none focus:ring-1 focus:ring-amber-500 shadow-sm"/>
                    </div>
                    <div class="flex items-end pt-5">
                        <button type="submit" id="btnAsk"
                                class="bg-amber-600 hover:bg-amber-700 text-white font-medium text-xs py-2 px-5 rounded-lg shadow-sm transition flex items-center gap-1.5">
                            <i class="fa-solid fa-magnifying-glass"></i> 질문하기
                        </button>
                    </div>
                </div>
            </form>

            <!-- B단계 4개 스텝 카드 그리드 -->
            <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                
                <!-- 스텝 B1 -->
                <div class="step-card bg-slate-50 border border-slate-200 rounded-xl p-4 flex flex-col justify-between">
                    <div>
                        <div class="flex items-center justify-between mb-2">
                            <span class="text-xs font-bold text-amber-700">스텝 B1</span>
                            <span id="badgeB1" class="text-[10px] px-2 py-0.5 rounded bg-slate-200 text-slate-600 font-medium">대기</span>
                        </div>
                        <h3 class="text-sm font-bold text-slate-800 mb-1">질문 입력</h3>
                        <p class="text-[11px] text-slate-500 mb-3">질문도 임베딩 벡터화</p>
                        <div id="outputB1" class="text-[11px] text-slate-600 space-y-1 bg-white p-2.5 rounded-lg border border-slate-100 min-h-[90px]">
                            <div class="text-slate-400 text-center py-5">질문을 입력해 주세요</div>
                        </div>
                    </div>
                    <div id="timeB1" class="text-[10px] text-slate-400 text-right mt-2">-</div>
                </div>

                <!-- 스텝 B2 -->
                <div class="step-card bg-slate-50 border border-slate-200 rounded-xl p-4 flex flex-col justify-between">
                    <div>
                        <div class="flex items-center justify-between mb-2">
                            <span class="text-xs font-bold text-amber-700">스텝 B2</span>
                            <span id="badgeB2" class="text-[10px] px-2 py-0.5 rounded bg-slate-200 text-slate-600 font-medium">대기</span>
                        </div>
                        <h3 class="text-sm font-bold text-slate-800 mb-1">관련 청크 검색</h3>
                        <p class="text-[11px] text-slate-500 mb-3">코사인 유사도 매칭</p>
                        <div id="outputB2" class="text-[11px] text-slate-600 space-y-1 bg-white p-2.5 rounded-lg border border-slate-100 min-h-[90px] max-h-[140px] overflow-y-auto">
                            <div class="text-slate-400 text-center py-5">유사도 검색 대기</div>
                        </div>
                    </div>
                    <div id="timeB2" class="text-[10px] text-slate-400 text-right mt-2">-</div>
                </div>

                <!-- 스텝 B3 -->
                <div class="step-card bg-slate-50 border border-slate-200 rounded-xl p-4 flex flex-col justify-between">
                    <div>
                        <div class="flex items-center justify-between mb-2">
                            <span class="text-xs font-bold text-amber-700">스텝 B3</span>
                            <span id="badgeB3" class="text-[10px] px-2 py-0.5 rounded bg-slate-200 text-slate-600 font-medium">대기</span>
                        </div>
                        <h3 class="text-sm font-bold text-slate-800 mb-1">프롬프트 구성</h3>
                        <p class="text-[11px] text-slate-500 mb-3">질문 + 원문 + 지시문 결합</p>
                        <div id="outputB3" class="text-[11px] text-slate-600 space-y-1 bg-white p-2.5 rounded-lg border border-slate-100 min-h-[90px]">
                            <div class="text-slate-400 text-center py-5">프롬프트 결합 대기</div>
                        </div>
                    </div>
                    <div id="timeB3" class="text-[10px] text-slate-400 text-right mt-2">-</div>
                </div>

                <!-- 스텝 B4 -->
                <div class="step-card bg-slate-50 border border-slate-200 rounded-xl p-4 flex flex-col justify-between">
                    <div>
                        <div class="flex items-center justify-between mb-2">
                            <span class="text-xs font-bold text-amber-700">스텝 B4</span>
                            <span id="badgeB4" class="text-[10px] px-2 py-0.5 rounded bg-slate-200 text-slate-600 font-medium">대기</span>
                        </div>
                        <h3 class="text-sm font-bold text-slate-800 mb-1">LLM 생성</h3>
                        <p class="text-[11px] text-slate-500 mb-3">근거를 읽고 답변 생성</p>
                        <div id="outputB4" class="text-[11px] text-slate-600 space-y-1 bg-white p-2.5 rounded-lg border border-slate-100 min-h-[90px]">
                            <div class="text-slate-400 text-center py-5">LLM 생성 대기</div>
                        </div>
                    </div>
                    <div id="timeB4" class="text-[10px] text-slate-400 text-right mt-2">-</div>
                </div>

            </div>

            <!-- 최종 생성된 답변 표시 박스 -->
            <div class="bg-gradient-to-r from-amber-50 to-orange-50 border border-amber-200 rounded-xl p-5">
                <div class="flex items-center justify-between mb-2">
                    <span class="text-xs font-bold text-amber-900 flex items-center gap-1.5">
                        <i class="fa-solid fa-robot text-amber-600"></i> 최종 생성된 AI 답변
                    </span>
                    <span id="answerLatency" class="text-xs text-amber-700"></span>
                </div>
                <div id="finalAnswerBox" class="bg-white/80 border border-amber-100 rounded-lg p-4 text-xs leading-relaxed text-slate-800 whitespace-pre-wrap min-h-[80px]">
                    질문을 입력하고 실행하면 근거 문서 기반의 답변이 여기에 표시됩니다.
                </div>
            </div>

        </section>

    </main>

    <!-- ===================================================================== -->
    <!-- 페이지 전체 내용 확인 & 페이지 이동 뷰어 모달 -->
    <!-- ===================================================================== -->
    <div id="pageViewerModal" class="hidden fixed inset-0 z-50 bg-slate-900/60 backdrop-blur-sm flex items-center justify-center p-4">
        <div class="bg-white rounded-2xl shadow-2xl w-full max-w-4xl max-h-[90vh] flex flex-col overflow-hidden border border-slate-200 animate-in fade-in duration-200">
            
            <!-- 모달 상단 헤더 -->
            <div class="px-6 py-4 bg-slate-50 border-b border-slate-200 flex flex-wrap items-center justify-between gap-3">
                <div class="flex items-center space-x-2">
                    <div class="w-8 h-8 rounded-lg bg-emerald-600 text-white flex items-center justify-center text-sm shadow">
                        <i class="fa-solid fa-file-lines"></i>
                    </div>
                    <div>
                        <h3 class="text-sm font-bold text-slate-900 flex items-center gap-2">
                            <span id="modalDocName">문서 제목</span>
                            <span class="text-[11px] bg-emerald-100 text-emerald-800 px-2 py-0.5 rounded-full font-medium">Gemini 3.8 Flash OCR</span>
                        </h3>
                        <p class="text-xs text-slate-500">순수 텍스트(활자) 내용 · <span id="modalDocStats">총 0페이지</span></p>
                    </div>
                </div>

                <!-- 페이지 이동 네비게이터 -->
                <div class="flex items-center gap-2">
                    <button type="button" onclick="navigatePage(-1)" 
                            class="px-3 py-1.5 text-xs bg-white border border-slate-300 hover:bg-slate-100 rounded-lg text-slate-700 font-medium transition flex items-center gap-1 shadow-sm">
                        <i class="fa-solid fa-chevron-left text-[10px]"></i> 이전 페이지
                    </button>
                    
                    <div class="flex items-center gap-1 text-xs">
                        <select id="modalPageSelect" onchange="onPageSelectChange(event)" 
                                class="px-2 py-1.5 text-xs bg-white border border-slate-300 rounded-lg font-semibold text-slate-800 focus:outline-none focus:ring-1 focus:ring-emerald-500 cursor-pointer shadow-sm">
                        </select>
                    </div>

                    <button type="button" onclick="navigatePage(1)" 
                            class="px-3 py-1.5 text-xs bg-white border border-slate-300 hover:bg-slate-100 rounded-lg text-slate-700 font-medium transition flex items-center gap-1 shadow-sm">
                        다음 페이지 <i class="fa-solid fa-chevron-right text-[10px]"></i>
                    </button>

                    <button type="button" onclick="copyCurrentPageText()" 
                            class="px-3 py-1.5 text-xs bg-emerald-50 hover:bg-emerald-100 border border-emerald-200 text-emerald-700 rounded-lg font-medium transition flex items-center gap-1">
                        <i class="fa-regular fa-copy"></i> 복사
                    </button>

                    <button type="button" onclick="closePageModal()" 
                            class="w-8 h-8 rounded-lg hover:bg-slate-200 text-slate-400 hover:text-slate-700 flex items-center justify-center transition">
                        <i class="fa-solid fa-xmark text-base"></i>
                    </button>
                </div>
            </div>

            <!-- 모달 본문: 해당 페이지 전체 내용 (순수 원문만 집중 열람) -->
            <div class="p-6 flex-1 overflow-y-auto bg-slate-100/50">
                <div class="flex items-center justify-between mb-2">
                    <span class="text-xs font-semibold text-slate-500">
                        <i class="fa-solid fa-bookmark text-emerald-600 mr-1"></i>
                        현재 위치: <span id="modalCurrentPageBadge" class="text-emerald-700 font-bold">1 페이지</span>
                    </span>
                    <span id="modalPageCharCount" class="text-xs text-slate-400 font-medium">0자</span>
                </div>
                <div id="modalPageTextBody" 
                     class="w-full bg-white border border-slate-200 rounded-xl p-5 text-slate-800 text-xs sm:text-sm leading-relaxed whitespace-pre-wrap font-mono min-h-[380px] max-h-[65vh] overflow-y-auto shadow-inner select-text">
                    (내용 로딩 중...)
                </div>
            </div>

            <!-- 모달 푸터 -->
            <div class="px-6 py-3 bg-white border-t border-slate-200 flex items-center justify-between text-xs text-slate-400">
                <div>
                    <i class="fa-regular fa-keyboard mr-1"></i> 좌우 방향키(←, →)로 페이지를 이동할 수 있습니다. (닫기: ESC)
                </div>
                <button type="button" onclick="closePageModal()" class="px-4 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg font-medium transition">
                    닫기
                </button>
            </div>

        </div>
    </div>

    <!-- ===================================================================== -->
    <!-- 스텝 A2 분할 청크 전체 문단 확인 모달 -->
    <!-- ===================================================================== -->
    <div id="chunksViewerModal" class="hidden fixed inset-0 z-50 bg-slate-900/60 backdrop-blur-sm flex items-center justify-center p-4">
        <div class="bg-white rounded-2xl shadow-2xl w-full max-w-4xl max-h-[90vh] flex flex-col overflow-hidden border border-slate-200 animate-in fade-in duration-200">
            <div class="px-6 py-4 bg-slate-50 border-b border-slate-200 flex items-center justify-between gap-3">
                <div class="flex items-center space-x-2.5">
                    <div class="w-8 h-8 rounded-lg bg-emerald-700 text-white flex items-center justify-center text-sm shadow">
                        <i class="fa-solid fa-scissors"></i>
                    </div>
                    <div>
                        <h3 class="text-sm font-bold text-slate-900 flex items-center gap-2">
                            <span>스텝 A2 전체 분할 청크 문단 목록</span>
                            <span id="chunksModalBadge" class="text-[11px] bg-emerald-100 text-emerald-800 px-2 py-0.5 rounded-full font-semibold">총 0개 청크</span>
                        </h3>
                        <p class="text-xs text-slate-500">페이지별 분할된 청크 문단 및 글자 수</p>
                    </div>
                </div>
                <div class="flex items-center gap-2">
                    <select id="chunksModalPageFilter" onchange="filterChunksModalPage(event)" 
                            class="px-2.5 py-1 text-xs bg-white border border-slate-300 rounded-lg font-semibold text-slate-800 focus:outline-none focus:ring-1 focus:ring-emerald-500 cursor-pointer shadow-sm">
                        <option value="all">전체 페이지 분할 청크 보기</option>
                    </select>
                    <button type="button" onclick="closeChunksModal()" class="w-8 h-8 rounded-lg hover:bg-slate-200 text-slate-400 hover:text-slate-700 flex items-center justify-center transition">
                        <i class="fa-solid fa-xmark text-base"></i>
                    </button>
                </div>
            </div>
            <div id="chunksModalBody" class="p-6 flex-1 overflow-y-auto bg-slate-100/50 space-y-4">
                <!-- 전체 청크 목록 -->
            </div>
            <div class="px-6 py-3 bg-white border-t border-slate-200 flex items-center justify-between text-xs text-slate-400">
                <div>각 청크의 복사 버튼으로 문단을 바로 복사할 수 있습니다. (닫기: ESC)</div>
                <button type="button" onclick="closeChunksModal()" class="px-4 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg font-medium transition">
                    닫기
                </button>
            </div>
        </div>
    </div>

    <script>
        // 전역 문서 페이지 상태
        let currentDocData = {
            fileName: '',
            totalPages: 0,
            totalChars: 0,
            pages: [],
            all_chunks: [],
            currentCardPage: 1,
            currentModalPage: 1
        };

        // HTML 특수문자 이스케이프 헬퍼
        function escapeHtml(str) {
            if (!str) return '';
            return String(str)
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }

        // 특정 청크 텍스트 복사 헬퍼
        async function copyChunkText(idx) {
            const el = document.getElementById(`chunkText_${idx}`);
            if (!el) return;
            try {
                await navigator.clipboard.writeText(el.innerText);
                const btn = event.currentTarget || event.target.closest('button');
                if (btn) {
                    const orig = btn.innerHTML;
                    btn.innerHTML = '<i class="fa-solid fa-check text-emerald-600"></i> 복사됨!';
                    setTimeout(() => { btn.innerHTML = orig; }, 1500);
                }
            } catch (err) {
                alert('복사 실패: ' + err);
            }
        }

        // 기본 OCR 프롬프트 저장 변수
        let defaultOcrPromptText = "";

        // OCR 프롬프트 글자 수 카운터 업데이트
        function updateOcrPromptCharCount() {
            const el = document.getElementById('ocrPromptInput');
            const cnt = document.getElementById('ocrPromptCharCount');
            if (el && cnt) {
                cnt.innerText = `${el.value.length.toLocaleString()}자`;
            }
        }

        // OCR 프롬프트 영역 접기/펼치기 토글
        function toggleOcrPromptArea() {
            const container = document.getElementById('ocrPromptContainer');
            const icon = document.getElementById('ocrPromptToggleIcon');
            if (!container || !icon) return;
            if (container.classList.contains('hidden')) {
                container.classList.remove('hidden');
                icon.classList.remove('fa-chevron-right');
                icon.classList.add('fa-chevron-down');
            } else {
                container.classList.add('hidden');
                icon.classList.remove('fa-chevron-down');
                icon.classList.add('fa-chevron-right');
            }
        }

        // 기본 OCR 프롬프트로 복원
        function resetOcrPrompt() {
            if (!defaultOcrPromptText) return;
            if (confirm('Gemini OCR 인식 프롬프트를 기본 권장 설정값으로 복원하시겠습니까?')) {
                const el = document.getElementById('ocrPromptInput');
                if (el) {
                    el.value = defaultOcrPromptText;
                    updateOcrPromptCharCount();
                }
            }
        }

        // 초기 상태 확인 및 문서 페이지 로드
        async function updateStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                document.getElementById('chunkCount').innerText = data.total_chunks_in_db;

                if (data.default_ocr_prompt) {
                    defaultOcrPromptText = data.default_ocr_prompt;
                }
                const promptEl = document.getElementById('ocrPromptInput');
                if (promptEl && (!promptEl.value || promptEl.value.trim() === '')) {
                    promptEl.value = (data.config && data.config.ocr_prompt) ? data.config.ocr_prompt : (data.default_ocr_prompt || '');
                    updateOcrPromptCharCount();
                }

                if (data.total_chunks_in_db > 0) {
                    await loadDocumentPages();
                }
            } catch (e) {
                console.error(e);
            }
        }
        updateStatus();

        // 서버로부터 인덱싱된 문서 페이지 목록 불러오기 (새로고침 시에도 유지)
        async function loadDocumentPages() {
            try {
                const res = await fetch('/api/document-pages');
                const data = await res.json();
                if (data.available && data.pages && data.pages.length > 0) {
                    currentDocData.fileName = data.file_name;
                    currentDocData.totalPages = data.total_pages;
                    currentDocData.totalChars = data.total_chars;
                    currentDocData.pages = data.pages;
                    currentDocData.all_chunks = data.all_chunks || [];
                    currentDocData.currentCardPage = 1;
                    currentDocData.currentModalPage = 1;

                    // 페이지별 분할 청크 매핑
                    currentDocData.pages.forEach(p => {
                        if (!p.chunks || !p.chunks.length) {
                            p.chunks = currentDocData.all_chunks.filter(c => c.page === p.page);
                        }
                        p.chunk_count = p.chunks ? p.chunks.length : 0;
                    });

                    renderA1Card();
                    renderA2Card();
                    document.getElementById('badgeA1').className = 'text-[10px] px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 font-medium';
                    document.getElementById('badgeA1').innerText = '완료';
                }
            } catch (err) {
                console.error('문서 페이지 로드 실패:', err);
            }
        }

        // A1 카드 내용 렌더링 (순수 원문만 페이지별로 표시)
        function renderA1Card() {
            if (!currentDocData.pages.length) return;

            const pIdx = Math.max(1, Math.min(currentDocData.currentCardPage, currentDocData.pages.length));
            currentDocData.currentCardPage = pIdx;
            const pageItem = currentDocData.pages[pIdx - 1];

            const pageText = pageItem.text
                ? escapeHtml(pageItem.text)
                : '<span class="text-slate-400 italic">(이 페이지에는 추출된 활자 텍스트가 없습니다 - 이미지/사진 전용 페이지)</span>';

            document.getElementById('outputA1').innerHTML = `
                <div class="flex items-center justify-between pb-1 border-b border-slate-100">
                    <span class="font-bold text-slate-800 truncate" title="${currentDocData.fileName}">${currentDocData.fileName}</span>
                    <span class="text-[10px] bg-emerald-50 text-emerald-700 border border-emerald-100 px-1.5 py-0.5 rounded font-semibold">${currentDocData.totalPages}p (${currentDocData.totalChars.toLocaleString()}자)</span>
                </div>
                <div class="mt-1 flex items-center justify-between text-[11px] text-emerald-900 font-medium bg-emerald-50/80 px-2 py-0.5 rounded">
                    <span><i class="fa-solid fa-file-lines mr-1 text-emerald-600"></i>${pageItem.page} 페이지 원문</span>
                    <span class="text-[10px] text-slate-500 font-normal">글자수: <b>${pageItem.char_count.toLocaleString()}자</b></span>
                </div>
                <div class="text-[11px] text-slate-700 bg-slate-50 p-2.5 rounded-lg border border-slate-200 min-h-[175px] max-h-[210px] overflow-y-auto whitespace-pre-wrap font-mono leading-relaxed select-text mt-1 shadow-inner">
${pageText}
                </div>
            `;

            const nav = document.getElementById('pageNavMiniA1');
            if (nav) {
                nav.classList.remove('hidden');
                nav.classList.add('flex');
            }
            const indicator = document.getElementById('cardPageIndicator');
            if (indicator) {
                indicator.innerText = `${pIdx} / ${currentDocData.totalPages}p`;
            }
        }

        // A1 카드 내에서 페이지 이동
        function navigateCardPage(delta) {
            if (!currentDocData.pages.length) return;
            const nextP = currentDocData.currentCardPage + delta;
            if (nextP < 1 || nextP > currentDocData.pages.length) return;
            currentDocData.currentCardPage = nextP;
            renderA1Card();
        }

        // A2 카드 내용 렌더링 (분할 청크 전체 요약 및 문단 리스트)
        function renderA2Card(stepA2Data) {
            const chunks = (stepA2Data && stepA2Data.chunks) || currentDocData.all_chunks || [];
            if (!chunks.length) {
                document.getElementById('outputA2').innerHTML = '<div class="text-slate-400 text-center py-5">청크 분할 결과 대기</div>';
                const chunkBtn = document.getElementById('chunkNavMiniA2');
                if (chunkBtn) chunkBtn.classList.add('hidden');
                return;
            }

            const chunkCount = stepA2Data ? stepA2Data.chunk_count : chunks.length;
            const lengths = chunks.map(c => c.char_count || (c.content ? c.content.length : 0));
            const minLen = stepA2Data ? stepA2Data.min_len : Math.min(...lengths);
            const maxLen = stepA2Data ? stepA2Data.max_len : Math.max(...lengths);
            const avgLen = stepA2Data ? stepA2Data.avg_len : Math.round(lengths.reduce((a, b) => a + b, 0) / lengths.length);

            let chunksHtml = `
                <div class="flex items-center justify-between pb-1 border-b border-slate-100">
                    <span class="font-bold text-slate-800">분할 청크: <b class="text-emerald-700">${chunkCount}개</b></span>
                    <span class="text-[10px] bg-emerald-50 text-emerald-700 border border-emerald-100 px-1.5 py-0.5 rounded font-semibold">
                        ${minLen} ~ ${maxLen}자 (평균 ${avgLen}자)
                    </span>
                </div>
                <div class="mt-1.5 text-[11px] text-slate-700 font-semibold flex items-center justify-between">
                    <span><i class="fa-solid fa-scissors text-emerald-600 mr-1"></i>나눠진 청크 문단 목록</span>
                    <span class="text-[10px] text-slate-400 font-normal">스크롤하여 확인</span>
                </div>
                <div class="mt-1 space-y-1.5 min-h-[175px] max-h-[210px] overflow-y-auto pr-1">
            `;

            chunks.forEach((c, idx) => {
                const cNum = (c.chunk_index !== undefined ? c.chunk_index : idx) + 1;
                const cLen = c.char_count || (c.content ? c.content.length : 0);
                const pNum = c.page || '?';
                const contentSafe = escapeHtml(c.content || '');
                chunksHtml += `
                    <div class="bg-white p-2 rounded-lg border border-slate-200 text-[10.5px] shadow-xs">
                        <div class="flex items-center justify-between font-semibold text-emerald-800 mb-0.5 border-b border-slate-100 pb-0.5">
                            <span class="flex items-center gap-1">
                                <span class="bg-emerald-100 text-emerald-800 px-1.5 py-0.2 rounded text-[10px] font-bold">#${cNum}</span>
                                <span>Page ${pNum}</span>
                            </span>
                            <span class="text-[10px] text-slate-500 font-normal">${cLen.toLocaleString()}자</span>
                        </div>
                        <div class="text-slate-700 whitespace-pre-wrap font-mono leading-relaxed line-clamp-3 select-text hover:line-clamp-none transition-all">
${contentSafe}
                        </div>
                    </div>
                `;
            });

            chunksHtml += `</div>`;
            document.getElementById('outputA2').innerHTML = chunksHtml;

            const nav = document.getElementById('chunkNavMiniA2');
            if (nav) nav.classList.remove('hidden');
            document.getElementById('badgeA2').className = 'text-[10px] px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 font-medium';
            document.getElementById('badgeA2').innerText = '완료';
        }

        // 모달 열기 (A1 전체 크게 보기)
        function openPageModal(targetPage) {
            if (!currentDocData.pages.length) {
                alert('확인할 문서 내용이 없습니다. 먼저 PDF를 인덱싱해 주세요.');
                return;
            }

            const pNum = targetPage || currentDocData.currentCardPage || 1;
            currentDocData.currentModalPage = Math.max(1, Math.min(pNum, currentDocData.pages.length));

            document.getElementById('modalDocName').innerText = currentDocData.fileName;
            document.getElementById('modalDocStats').innerText = `총 ${currentDocData.totalPages}페이지 · ${currentDocData.totalChars.toLocaleString()}자`;

            // 셀렉트 박스 옵션 채우기
            const sel = document.getElementById('modalPageSelect');
            sel.innerHTML = '';
            currentDocData.pages.forEach(p => {
                const opt = document.createElement('option');
                opt.value = p.page;
                opt.innerText = `${p.page}페이지 (${p.char_count.toLocaleString()}자)`;
                if (p.page === currentDocData.currentModalPage) opt.selected = true;
                sel.appendChild(opt);
            });

            renderModalPage(currentDocData.currentModalPage);

            const modal = document.getElementById('pageViewerModal');
            modal.classList.remove('hidden');
            modal.classList.add('flex');
        }

        // 모달 내 페이지 렌더링
        function renderModalPage(pageNum) {
            const pageItem = currentDocData.pages.find(p => p.page === pageNum) || currentDocData.pages[pageNum - 1];
            if (!pageItem) return;

            currentDocData.currentModalPage = pageItem.page;

            document.getElementById('modalCurrentPageBadge').innerText = `${pageItem.page} / ${currentDocData.totalPages} 페이지`;
            document.getElementById('modalPageCharCount').innerText = `원문 글자 수: ${pageItem.char_count.toLocaleString()}자`;
            document.getElementById('modalPageTextBody').innerText = pageItem.text || '(이 페이지에는 추출된 활자 텍스트가 없습니다 - 이미지/사진 전용 페이지)';

            const sel = document.getElementById('modalPageSelect');
            if (sel) sel.value = pageItem.page;
        }

        // 모달 내 페이지 이동 (이전/다음)
        function navigatePage(delta) {
            if (!currentDocData.pages.length) return;
            const nextP = currentDocData.currentModalPage + delta;
            if (nextP < 1 || nextP > currentDocData.pages.length) return;
            renderModalPage(nextP);
        }

        // 셀렉트 박스 변경 시
        function onPageSelectChange(e) {
            const pageNum = parseInt(e.target.value);
            renderModalPage(pageNum);
        }

        // 모달 닫기
        function closePageModal() {
            const modal = document.getElementById('pageViewerModal');
            modal.classList.add('hidden');
            modal.classList.remove('flex');
        }

        // 스텝 A2 전체 분할 청크 모달 열기
        function openChunksModal() {
            const chunks = currentDocData.all_chunks || [];
            if (!chunks.length) {
                alert('분할된 청크 데이터가 없습니다. 먼저 PDF 문서를 인덱싱해 주세요.');
                return;
            }

            const modalBadge = document.getElementById('chunksModalBadge');
            if (modalBadge) {
                modalBadge.innerText = `총 ${chunks.length}개 청크 (${currentDocData.totalPages}페이지 분할)`;
            }

            // 페이지 셀렉트 필터 채우기
            const filterSel = document.getElementById('chunksModalPageFilter');
            if (filterSel) {
                const uniquePages = [...new Set(chunks.map(c => c.page || 1))].sort((a, b) => a - b);
                filterSel.innerHTML = '<option value="all">전체 페이지 분할 청크 보기 (총 ' + chunks.length + '개)</option>';
                uniquePages.forEach(p => {
                    const pChunks = chunks.filter(c => c.page === p);
                    const opt = document.createElement('option');
                    opt.value = p;
                    opt.innerText = `${p}페이지 분할 청크 (${pChunks.length}개)`;
                    filterSel.appendChild(opt);
                });
            }

            renderChunksModalBody('all');

            const modal = document.getElementById('chunksViewerModal');
            if (modal) {
                modal.classList.remove('hidden');
                modal.classList.add('flex');
            }
        }

        // 페이지별 청크 필터링 렌더러
        function renderChunksModalBody(selectedPage) {
            const chunks = currentDocData.all_chunks || [];
            const bodyEl = document.getElementById('chunksModalBody');
            if (!bodyEl) return;

            // 페이지별 그룹화
            const chunksByPage = {};
            chunks.forEach(c => {
                const p = c.page || 1;
                if (selectedPage !== 'all' && String(p) !== String(selectedPage)) return;
                if (!chunksByPage[p]) chunksByPage[p] = [];
                chunksByPage[p].push(c);
            });

            let html = '';
            const sortedPages = Object.keys(chunksByPage).map(Number).sort((a, b) => a - b);
            if (!sortedPages.length) {
                bodyEl.innerHTML = '<div class="text-xs text-slate-400 p-8 text-center bg-white rounded-xl border">선택한 페이지에 해당하는 분할 청크가 없습니다.</div>';
                return;
            }

            sortedPages.forEach(p => {
                const pageChunks = chunksByPage[p];
                const pageChars = pageChunks.reduce((acc, c) => acc + (c.char_count || (c.content ? c.content.length : 0)), 0);
                html += `
                    <div class="bg-white rounded-xl border border-slate-200 p-4 shadow-xs">
                        <div class="flex items-center justify-between pb-2 mb-3 border-b border-slate-100">
                            <div class="flex items-center gap-2">
                                <span class="w-6 h-6 rounded-md bg-emerald-700 text-white flex items-center justify-center text-xs font-bold">${p}</span>
                                <h4 class="text-xs font-bold text-slate-800">${p}페이지 분할 청크 (${pageChunks.length}개)</h4>
                            </div>
                            <span class="text-xs text-slate-500 font-medium">청크 합계: <b>${pageChars.toLocaleString()}자</b></span>
                        </div>
                        <div class="space-y-3">
                `;

                pageChunks.forEach((c, idx) => {
                    const cNum = (c.chunk_index !== undefined ? c.chunk_index : idx) + 1;
                    const cLen = c.char_count || (c.content ? c.content.length : 0);
                    html += `
                        <div class="bg-slate-50/70 p-3 rounded-lg border border-slate-200">
                            <div class="flex items-center justify-between pb-1.5 mb-1.5 border-b border-slate-200">
                                <span class="font-bold text-xs text-emerald-900">청크 #${cNum} <span class="text-slate-400 font-normal text-[11px]">(Page ${p})</span></span>
                                <div class="flex items-center gap-2">
                                    <span class="text-[11px] font-semibold text-slate-600 bg-white border border-slate-200 px-2 py-0.5 rounded font-mono">${cLen.toLocaleString()}자</span>
                                    <button type="button" onclick="copyChunkText(${c.chunk_index !== undefined ? c.chunk_index : idx})" class="text-[11px] text-slate-500 hover:text-emerald-700 px-2 py-0.5 border bg-white rounded hover:bg-slate-100 cursor-pointer">
                                        <i class="fa-regular fa-copy mr-0.5"></i> 복사
                                    </button>
                                </div>
                            </div>
                            <div id="chunkText_${c.chunk_index !== undefined ? c.chunk_index : idx}" class="text-xs text-slate-800 whitespace-pre-wrap font-mono leading-relaxed bg-white p-2.5 rounded border border-slate-200 select-text">
${escapeHtml(c.content)}
                            </div>
                        </div>
                    `;
                });

                html += `</div></div>`;
            });

            bodyEl.innerHTML = html;
        }

        // 모달 상단 페이지 필터 변경 핸들러
        function filterChunksModalPage(e) {
            renderChunksModalBody(e.target.value);
        }

        // 스텝 A2 전체 분할 청크 모달 닫기
        function closeChunksModal() {
            const modal = document.getElementById('chunksViewerModal');
            if (modal) {
                modal.classList.add('hidden');
                modal.classList.remove('flex');
            }
        }

        // 현재 페이지 텍스트 클립보드 복사
        async function copyCurrentPageText() {
            const text = document.getElementById('modalPageTextBody').innerText;
            if (!text) return;
            try {
                await navigator.clipboard.writeText(text);
                const btn = event.currentTarget || event.target.closest('button');
                const orig = btn.innerHTML;
                btn.innerHTML = '<i class="fa-solid fa-check text-emerald-600"></i> 복사 완료!';
                setTimeout(() => { btn.innerHTML = orig; }, 1500);
            } catch (err) {
                alert('복사 실패: ' + err);
            }
        }

        // 키보드 네비게이션 이벤트
        window.addEventListener('keydown', (e) => {
            const pModal = document.getElementById('pageViewerModal');
            const cModal = document.getElementById('chunksViewerModal');

            if (e.key === 'Escape') {
                if (pModal && !pModal.classList.contains('hidden')) {
                    e.preventDefault();
                    closePageModal();
                    return;
                }
                if (cModal && !cModal.classList.contains('hidden')) {
                    e.preventDefault();
                    closeChunksModal();
                    return;
                }
            }

            if (pModal && !pModal.classList.contains('hidden')) {
                if (e.key === 'ArrowLeft') {
                    e.preventDefault();
                    navigatePage(-1);
                } else if (e.key === 'ArrowRight') {
                    e.preventDefault();
                    navigatePage(1);
                }
            }
        });

        // DB 초기화
        async function clearDb() {
            if (!confirm('정말로 SQLite 벡터 DB를 초기화하시겠습니까?')) return;
            try {
                const res = await fetch('/api/clear', { method: 'DELETE' });
                const data = await res.json();
                alert(data.message);
                currentDocData = { fileName: '', totalPages: 0, totalChars: 0, pages: [], all_chunks: [], currentCardPage: 1, currentModalPage: 1 };
                const pPanel = document.getElementById('indexingProgressPanel');
                if (pPanel) pPanel.classList.add('hidden');
                document.getElementById('outputA1').innerHTML = '<div class="text-slate-400 text-center py-5">PDF 문서를 업로드하세요</div>';
                document.getElementById('outputA2').innerHTML = '<div class="text-slate-400 text-center py-5">청크 분할 결과 대기</div>';
                const navA1 = document.getElementById('pageNavMiniA1');
                if (navA1) {
                    navA1.classList.add('hidden');
                    navA1.classList.remove('flex');
                }
                const navA2 = document.getElementById('chunkNavMiniA2');
                if (navA2) {
                    navA2.classList.add('hidden');
                }
                ['A1', 'A2', 'A3', 'A4'].forEach(id => {
                    const b = document.getElementById('badge' + id);
                    if (b) {
                        b.className = 'text-[10px] px-2 py-0.5 rounded bg-slate-200 text-slate-600 font-medium';
                        b.innerText = '대기';
                    }
                    const t = document.getElementById('time' + id);
                    if (t) t.innerText = '-';
                });
                updateStatus();
            } catch (e) {
                alert('초기화 실패: ' + e);
            }
        }

        // Gemini API 429 오류 정밀 진단 모달 팝업
        function showDiagnosisModal(diag) {
            let modal = document.getElementById('diagnosisModal');
            if (modal) modal.remove();
            
            modal = document.createElement('div');
            modal.id = 'diagnosisModal';
            modal.className = 'fixed inset-0 bg-slate-900/60 z-50 flex items-center justify-center p-4 backdrop-blur-xs';
            
            const isQuota = diag.category === 'PROJECT_QUOTA_EXCEEDED';
            const badgeColor = isQuota ? 'bg-rose-100 text-rose-800 border-rose-300' : 'bg-amber-100 text-amber-800 border-amber-300';
            const headerIcon = isQuota ? 'fa-solid fa-triangle-exclamation text-rose-600' : 'fa-solid fa-cloud-bolt text-amber-600';
            
            let metricsHtml = '';
            if (diag.quota_metrics && diag.quota_metrics.length > 0) {
                metricsHtml += `<div class="mb-2"><span class="font-bold text-slate-700">위반 메트릭:</span> <code class="bg-rose-50 text-rose-700 px-1.5 py-0.5 rounded text-[11px] font-mono border border-rose-200">${diag.quota_metrics.join(', ')}</code></div>`;
            }
            if (diag.retry_delay) {
                metricsHtml += `<div class="mb-2"><span class="font-bold text-slate-700">권장 재시도 대기:</span> <span class="bg-amber-100 text-amber-800 px-2 py-0.5 rounded font-bold text-[11px] border border-amber-200">${diag.retry_delay}</span></div>`;
            }
            if (diag.violations && diag.violations.length > 0) {
                metricsHtml += `<div class="mb-2 text-slate-600 text-[11px] bg-slate-50 p-2.5 rounded-lg border border-slate-200"><b>한도 세부내용:</b> ${diag.violations[0]}</div>`;
            }

            modal.innerHTML = `
                <div class="bg-white rounded-2xl max-w-lg w-full p-6 shadow-2xl border border-slate-200 animate-in fade-in zoom-in-95 duration-150">
                    <div class="flex items-start justify-between pb-3 mb-4 border-b border-slate-100">
                        <div class="flex items-center gap-2.5">
                            <div class="w-10 h-10 rounded-xl bg-slate-100 flex items-center justify-center text-lg">
                                <i class="${headerIcon}"></i>
                            </div>
                            <div>
                                <h3 class="text-base font-bold text-slate-900">Gemini API 429 오류 원인 진단</h3>
                                <p class="text-[11px] text-slate-400">Google Gemini API RESOURCE_EXHAUSTED</p>
                            </div>
                        </div>
                        <button onclick="document.getElementById('diagnosisModal').remove()" class="text-slate-400 hover:text-slate-600 text-sm p-1 rounded-lg">
                            <i class="fa-solid fa-xmark"></i>
                        </button>
                    </div>

                    <div class="space-y-3.5 text-xs">
                        <div>
                            <span class="inline-block px-2.5 py-1 rounded-full text-xs font-bold border ${badgeColor} mb-2">
                                ${diag.category_name}
                            </span>
                            <p class="text-slate-700 leading-relaxed font-medium bg-slate-50 p-3 rounded-xl border border-slate-200">
                                ${diag.root_cause.replace(/\\n/g, '<br>')}
                            </p>
                        </div>

                        ${metricsHtml}

                        <div class="bg-indigo-50/70 p-3.5 rounded-xl border border-indigo-100 text-indigo-950">
                            <div class="font-bold mb-1.5 flex items-center gap-1.5 text-indigo-900">
                                <i class="fa-solid fa-lightbulb text-amber-500"></i> 권장 조치 가이드 (Next Actions)
                            </div>
                            <div class="leading-relaxed space-y-1 text-[11px] text-slate-700">
                                ${diag.action_guide.split('\\n').map(l => `<div>${l}</div>`).join('')}
                            </div>
                        </div>
                    </div>

                    <div class="mt-5 flex justify-end">
                        <button onclick="document.getElementById('diagnosisModal').remove()" class="px-4 py-2 bg-slate-900 hover:bg-slate-800 text-white rounded-xl text-xs font-semibold shadow-xs transition">
                            확인 및 닫기
                        </button>
                    </div>
                </div>
            `;
            document.body.appendChild(modal);
        }

        // A단계: 실시간 SSE 스트리밍 인덱싱 제출 처리
        let progressTimerInterval = null;
        let progressStartTime = 0;

        function updateProgressTimer() {
            if (!progressStartTime) return;
            const elapsedSec = Math.floor((Date.now() - progressStartTime) / 1000);
            const m = String(Math.floor(elapsedSec / 60)).padStart(2, '0');
            const s = String(elapsedSec % 60).padStart(2, '0');
            const el = document.getElementById('progressElapsedTime');
            if (el) el.innerText = `⏱ ${m}:${s}`;
        }

        function setStepIndicatorState(stepId, state) {
            // state: 'waiting' | 'active' | 'done'
            const el = document.getElementById('stepIndicator' + stepId);
            if (!el) return;
            if (state === 'active') {
                el.className = 'flex items-center justify-center gap-1.5 py-1.5 px-2 rounded-xl text-xs transition-all border bg-emerald-500/20 text-emerald-300 border-emerald-400/60 ring-2 ring-emerald-400/30 animate-pulse font-bold shadow-xs';
            } else if (state === 'done') {
                el.className = 'flex items-center justify-center gap-1.5 py-1.5 px-2 rounded-xl text-xs transition-all border bg-emerald-600/90 text-white border-emerald-400/40 font-bold shadow-xs';
            } else {
                el.className = 'flex items-center justify-center gap-1.5 py-1.5 px-2 rounded-xl text-xs transition-all border bg-white/5 border-white/10 text-slate-400 font-normal';
            }
        }

        async function handleIndex(e) {
            if (e) {
                e.preventDefault();
                if (typeof e.stopPropagation === 'function') e.stopPropagation();
            }

            console.log('[인덱싱 시작 버튼 클릭]');
            const fileInput = document.getElementById('pdfFileInput');
            if (!fileInput || !fileInput.files || !fileInput.files.length) {
                alert('인덱싱할 PDF 파일을 선택해 주세요.');
                return false;
            }

            const btn = document.getElementById('btnIndex');
            if (btn) {
                btn.disabled = true;
                btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> 인덱싱 진행 중...';
            }

            // 프로그레스 패널 활성화 및 초기화
            const progressPanel = document.getElementById('indexingProgressPanel');
            if (progressPanel) progressPanel.classList.remove('hidden');

            const pBar = document.getElementById('progressBarFill');
            const pText = document.getElementById('progressPercentText');
            const pBadge = document.getElementById('progressStepBadge');
            const pTextBody = document.getElementById('progressStatusText');
            const pDetail = document.getElementById('progressDetailMessage');
            const pIcon = document.getElementById('progressStatusIcon');

            if (pBar) {
                pBar.className = 'bg-gradient-to-r from-emerald-500 via-teal-400 to-emerald-300 h-full rounded-full transition-all duration-300 ease-out shadow-sm';
                pBar.style.width = '3%';
            }
            if (pText) pText.innerText = '3%';
            if (pBadge) {
                pBadge.className = 'px-2.5 py-1 rounded-lg bg-emerald-500 text-slate-950 font-bold text-xs tracking-wider uppercase shadow-xs';
                pBadge.innerText = '스텝 A1';
            }
            if (pTextBody) pTextBody.innerText = '인덱싱 준비 및 문서 분석 요청 중...';
            if (pDetail) pDetail.innerText = '업로드된 문서를 분석하기 위해 준비 작업을 진행하고 있습니다.';
            if (pIcon) pIcon.className = 'fa-solid fa-circle-notch fa-spin text-emerald-400';

            // 4개 미니 인디케이터 초기화
            setStepIndicatorState('A1', 'active');
            setStepIndicatorState('A2', 'waiting');
            setStepIndicatorState('A3', 'waiting');
            setStepIndicatorState('A4', 'waiting');

            // 4개 하단 카드 뱃지 및 내용 초기화 (이전 작업 내용 완전 리셋)
            const bA1 = document.getElementById('badgeA1');
            if (bA1) {
                bA1.className = 'text-[10px] px-2 py-0.5 rounded bg-amber-100 text-amber-800 font-medium animate-pulse';
                bA1.innerText = '처리 중';
            }
            ['A2', 'A3', 'A4'].forEach(id => {
                const b = document.getElementById('badge' + id);
                if (b) {
                    b.className = 'text-[10px] px-2 py-0.5 rounded bg-slate-200 text-slate-600 font-medium';
                    b.innerText = '대기';
                }
                const t = document.getElementById('time' + id);
                if (t) t.innerText = '-';
            });
            const tA1 = document.getElementById('timeA1');
            if (tA1) tA1.innerText = '-';

            const outA1 = document.getElementById('outputA1');
            if (outA1) outA1.innerHTML = '<div class="text-slate-400 text-center py-5"><i class="fa-solid fa-circle-notch fa-spin mr-1"></i> 문서 읽는 중...</div>';
            const outA2 = document.getElementById('outputA2');
            if (outA2) outA2.innerHTML = '<div class="text-slate-400 text-center py-5">청크 분할 대기 중</div>';
            const outA3 = document.getElementById('outputA3');
            if (outA3) outA3.innerHTML = '<div class="text-slate-400 text-center py-5">임베딩 대기 중</div>';
            const outA4 = document.getElementById('outputA4');
            if (outA4) outA4.innerHTML = '<div class="text-slate-400 text-center py-5">저장 대기 중</div>';

            // 타이머 시작
            if (progressTimerInterval) clearInterval(progressTimerInterval);
            progressStartTime = Date.now();
            updateProgressTimer();
            progressTimerInterval = setInterval(updateProgressTimer, 1000);

            const detectSubsetChecked = document.getElementById('detectSubsetCheckbox')?.checked ?? true;
            const forceReindexChecked = document.getElementById('forceReindexCheckbox')?.checked ?? false;

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            formData.append('chunk_size', document.getElementById('chunkSizeInput')?.value || '500');
            formData.append('chunk_overlap', document.getElementById('chunkOverlapInput')?.value || '100');
            formData.append('clear_existing', forceReindexChecked ? 'true' : 'false');
            formData.append('detect_subset', detectSubsetChecked ? 'true' : 'false');
            formData.append('force_reindex', forceReindexChecked ? 'true' : 'false');
            const ocrPromptVal = document.getElementById('ocrPromptInput')?.value?.trim();
            if (ocrPromptVal) {
                formData.append('ocr_prompt', ocrPromptVal);
            }

            try {
                // 1차 시도: SSE 실시간 스트리밍 엔드포인트 호출
                const response = await fetch('/api/index-stream', {
                    method: 'POST',
                    body: formData,
                });

                if (!response.ok) {
                    throw new Error(`스트리밍 연결 오류 (HTTP ${response.status})`);
                }

                const reader = response.body ? response.body.getReader() : null;
                if (!reader) {
                    throw new Error('브라우저가 스트리밍을 지원하지 않습니다. 기본 인덱싱으로 전환합니다.');
                }

                const decoder = new TextDecoder('utf-8');
                let buffer = '';
                let finalResult = null;

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const sNorm = buffer.replace(new RegExp(String.fromCharCode(13), 'g'), '');
                    const sep = String.fromCharCode(10) + String.fromCharCode(10);
                    const parts = sNorm.split(sep);
                    buffer = parts.pop() || ''; // 미완성 청크 유지

                    for (const block of parts) {
                        const trimmed = block.trim();
                        if (!trimmed || trimmed.startsWith(':')) continue; // keep-alive 무시

                        if (trimmed.startsWith('data: ')) {
                            const rawJson = trimmed.slice(6);
                            try {
                                const evt = JSON.parse(rawJson);

                                if (evt.type === 'progress') {
                                    // 1. 프로그레스 바 폭 및 퍼센트 수치 반영
                                    if (pBar && typeof evt.percent === 'number') {
                                        pBar.style.width = `${Math.min(100, Math.max(0, evt.percent))}%`;
                                    }
                                    if (pText && typeof evt.percent === 'number') {
                                        pText.innerText = `${evt.percent}%`;
                                    }
                                    if (pBadge && evt.step && evt.step !== 'A0') {
                                        pBadge.innerText = `스텝 ${evt.step}`;
                                    }
                                    if (pTextBody && evt.message) {
                                        pTextBody.innerText = evt.message;
                                    }

                                    // 2. 단계별 실시간 화면 렌더링
                                    if (evt.step === 'A1') {
                                        setStepIndicatorState('A1', 'active');
                                        if (evt.data) {
                                            const a1 = evt.data;
                                            currentDocData.fileName = a1.file_name;
                                            currentDocData.totalPages = a1.total_pages;
                                            currentDocData.totalChars = a1.total_chars;
                                            currentDocData.pages = a1.pages_content || [];
                                            currentDocData.currentCardPage = 1;
                                            currentDocData.currentModalPage = 1;
                                            renderA1Card();

                                            setStepIndicatorState('A1', 'done');
                                            setStepIndicatorState('A2', 'active');
                                            const b1 = document.getElementById('badgeA1');
                                            if (b1) {
                                                b1.className = 'text-[10px] px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 font-medium';
                                                b1.innerText = '완료';
                                            }
                                            const t1 = document.getElementById('timeA1');
                                            if (t1) t1.innerText = `${a1.elapsed}s`;

                                            const b2 = document.getElementById('badgeA2');
                                            if (b2) {
                                                b2.className = 'text-[10px] px-2 py-0.5 rounded bg-amber-100 text-amber-800 font-medium animate-pulse';
                                                b2.innerText = '처리 중';
                                            }
                                            if (pDetail) pDetail.innerText = `스텝 A1 완료: ${a1.total_pages}페이지(${a1.total_chars.toLocaleString()}자) 텍스트 추출 완료.`;
                                        }
                                    } else if (evt.step === 'A2') {
                                        setStepIndicatorState('A2', 'active');
                                        if (evt.data) {
                                            const a2 = evt.data;
                                            currentDocData.all_chunks = a2.chunks || [];
                                            currentDocData.pages.forEach(p => {
                                                p.chunks = currentDocData.all_chunks.filter(c => c.page === p.page);
                                                p.chunk_count = p.chunks.length;
                                            });
                                            renderA2Card(a2);

                                            setStepIndicatorState('A2', 'done');
                                            setStepIndicatorState('A3', 'active');
                                            const b2 = document.getElementById('badgeA2');
                                            if (b2) {
                                                b2.className = 'text-[10px] px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 font-medium';
                                                b2.innerText = '완료';
                                            }
                                            const t2 = document.getElementById('timeA2');
                                            if (t2) t2.innerText = `${a2.elapsed}s`;

                                            const b3 = document.getElementById('badgeA3');
                                            if (b3) {
                                                b3.className = 'text-[10px] px-2 py-0.5 rounded bg-amber-100 text-amber-800 font-medium animate-pulse';
                                                b3.innerText = '처리 중';
                                            }
                                            if (pDetail) pDetail.innerText = `스텝 A2 완료: 총 ${a2.chunk_count}개 청크 분할 완료 (평균 ${a2.avg_len}자).`;
                                        }
                                    } else if (evt.step === 'A3') {
                                        setStepIndicatorState('A3', 'active');
                                        if (evt.data) {
                                            const a3 = evt.data;
                                            const sampleVec = (a3.sample_vector || []).slice(0, 3).join(', ');
                                            const outA3 = document.getElementById('outputA3');
                                            const hitMsg = (a3.cache_hit_count && a3.cache_hit_count > 0)
                                                ? `<div class="text-[11px] text-emerald-600 font-semibold"><i class="fa-solid fa-bolt mr-1"></i> 캐시 적중: <b>${a3.cache_hit_count}개</b> 재사용 (비용 0원)</div>`
                                                : `<div class="text-[11px] text-slate-500">신규 생성: ${a3.vector_count}개</div>`;
                                            if (outA3) {
                                                outA3.innerHTML = `
                                                    <div>임베딩: <b class="text-emerald-700">${a3.vector_count}개</b> (${a3.dimension}차원)</div>
                                                    <div>모델: gemini-embedding-2</div>
                                                    ${hitMsg}
                                                    <div class="text-[10px] text-slate-400 font-mono truncate">[${sampleVec}, ...]</div>
                                                `;
                                            }

                                            setStepIndicatorState('A3', 'done');
                                            setStepIndicatorState('A4', 'active');
                                            const b3 = document.getElementById('badgeA3');
                                            if (b3) {
                                                b3.className = 'text-[10px] px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 font-medium';
                                                b3.innerText = '완료';
                                            }
                                            const t3 = document.getElementById('timeA3');
                                            if (t3) t3.innerText = `${a3.elapsed}s`;

                                            const b4 = document.getElementById('badgeA4');
                                            if (b4) {
                                                b4.className = 'text-[10px] px-2 py-0.5 rounded bg-amber-100 text-amber-800 font-medium animate-pulse';
                                                b4.innerText = '처리 중';
                                            }
                                            if (pDetail) pDetail.innerText = `스텝 A3 완료: ${a3.vector_count}개 청크 임베딩 생성 완료.`;
                                        }
                                    } else if (evt.step === 'A4') {
                                        setStepIndicatorState('A4', 'active');
                                        if (evt.data) {
                                            const a4 = evt.data;
                                            const outA4 = document.getElementById('outputA4');
                                            if (outA4) {
                                                outA4.innerHTML = `
                                                    <div>저장 레코드: <b>${a4.saved_count}개</b></div>
                                                    <div>DB 파일: ${a4.file_size_kb} KB</div>
                                                    <div class="text-[10px] text-emerald-600 font-semibold">SQLite BLOB + 출처 매핑 완료</div>
                                                `;
                                            }

                                            setStepIndicatorState('A4', 'done');
                                            const b4 = document.getElementById('badgeA4');
                                            if (b4) {
                                                b4.className = 'text-[10px] px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 font-medium';
                                                b4.innerText = '완료';
                                            }
                                            const t4 = document.getElementById('timeA4');
                                            if (t4) t4.innerText = `${a4.elapsed}s`;
                                            if (pDetail) pDetail.innerText = `스텝 A4 완료: ${a4.saved_count}개 레코드 SQLite DB 저장 완료.`;
                                        }
                                    }
                                } else if (evt.type === 'complete') {
                                    finalResult = evt.result;
                                    
                                    if (finalResult && finalResult.skipped) {
                                        // [하위 문서 또는 중복 파일 스킵 시]
                                        if (pBar) {
                                            pBar.className = 'bg-gradient-to-r from-amber-500 via-yellow-400 to-amber-300 h-full rounded-full transition-all duration-300 shadow-sm';
                                            pBar.style.width = '100%';
                                        }
                                        if (pText) pText.innerText = '100%';
                                        if (pBadge) {
                                            pBadge.className = 'px-2.5 py-1 rounded-lg bg-amber-400 text-slate-950 font-bold text-xs tracking-wider uppercase';
                                            pBadge.innerText = finalResult.skipped_as_subset ? '하위 문서 감지' : '중복 파일 감지';
                                        }
                                        if (pIcon) pIcon.className = 'fa-solid fa-shield-halved text-amber-300';
                                        if (pTextBody) pTextBody.innerText = finalResult.message;
                                        if (pDetail) pDetail.innerText = '기존 등록된 문서에 이미 포함된 내용이므로 불필요한 청킹/임베딩 API 비용 및 중복 DB 저장을 안전하게 건너뛰었습니다.';

                                        const alertBanner = document.getElementById('subsetAlertBanner');
                                        const alertText = document.getElementById('subsetAlertText');
                                        const alertTitle = document.getElementById('subsetAlertTitle');
                                        const ratioBadge = document.getElementById('subsetRatioBadge');
                                        if (alertBanner && alertText) {
                                            alertBanner.classList.remove('hidden');
                                            alertText.innerText = finalResult.message;
                                            if (finalResult.subset_info) {
                                                const ratioPct = Math.round(finalResult.subset_info.overlap_ratio * 100);
                                                if (ratioBadge) ratioBadge.innerText = `일치율 ${ratioPct}%`;
                                                if (alertTitle) alertTitle.innerText = `'${finalResult.subset_info.matched_file}'의 부분 발췌본(하위 문서) 감지`;
                                            }
                                        }

                                        setStepIndicatorState('A1', 'done');
                                        ['A2', 'A3', 'A4'].forEach(id => {
                                            setStepIndicatorState(id, 'waiting');
                                            const b = document.getElementById('badge' + id);
                                            if (b) {
                                                b.className = 'text-[10px] px-2 py-0.5 rounded bg-amber-100 text-amber-800 font-medium';
                                                b.innerText = '중복 스킵';
                                            }
                                        });

                                        const outA2 = document.getElementById('outputA2');
                                        if (outA2) outA2.innerHTML = '<div class="text-xs text-amber-700 font-medium py-3 text-center"><i class="fa-solid fa-filter mr-1"></i> 기존 등록 문서의 하위 발췌본으로 확인되어 청크 생성을 건너뛰었습니다.</div>';
                                        const outA3 = document.getElementById('outputA3');
                                        if (outA3) outA3.innerHTML = '<div class="text-xs text-amber-700 font-medium py-3 text-center"><i class="fa-solid fa-piggy-bank mr-1"></i> Gemini 임베딩 API 호출 비용 100% 절약 (스킵됨)</div>';
                                        const outA4 = document.getElementById('outputA4');
                                        if (outA4) outA4.innerHTML = '<div class="text-xs text-amber-700 font-medium py-3 text-center"><i class="fa-solid fa-database mr-1"></i> 기존 SQLite 벡터 DB 데이터 보존 (신규 적재 안 함)</div>';
                                    } else {
                                        // [정상 신규 인덱싱 완료 시]
                                        if (pBar) {
                                            pBar.className = 'bg-gradient-to-r from-emerald-500 via-teal-400 to-emerald-300 h-full rounded-full transition-all duration-300 ease-out shadow-sm';
                                            pBar.style.width = '100%';
                                        }
                                        if (pText) pText.innerText = '100%';
                                        if (pBadge) {
                                            pBadge.className = 'px-2.5 py-1 rounded-lg bg-emerald-400 text-slate-950 font-bold text-xs tracking-wider uppercase';
                                            pBadge.innerText = '전체 완료';
                                        }
                                        if (pIcon) pIcon.className = 'fa-solid fa-circle-check text-emerald-300';
                                        if (pTextBody) pTextBody.innerText = '인덱싱 파이프라인 전체 완료!';
                                        if (finalResult && finalResult.step_a1_load && finalResult.step_a2_split) {
                                            if (pDetail) pDetail.innerText = `총 ${finalResult.step_a1_load.total_pages}페이지 (${finalResult.step_a1_load.total_chars.toLocaleString()}자), ${finalResult.step_a2_split.chunk_count}개 청크가 성공적으로 SQLite 벡터 스토어에 등록되었습니다.`;
                                        }
                                        ['A1', 'A2', 'A3', 'A4'].forEach(id => setStepIndicatorState(id, 'done'));
                                    }
                                } else if (evt.type === 'error') {
                                    if (evt.diagnosis) {
                                        showDiagnosisModal(evt.diagnosis);
                                    }
                                    throw new Error(evt.error || '인덱싱 처리 중 오류가 발생했습니다.');
                                }
                            } catch (parseErr) {
                                console.error('SSE JSON 파싱 오류:', parseErr, rawJson);
                            }
                        }
                    }
                }

                if (finalResult) {
                    updateStatus();
                }
            } catch (err) {
                console.warn('[스트리밍 실패, 기본 동기식 인덱싱으로 자동 재시도합니다]', err);
                if (pTextBody) pTextBody.innerText = '기본 모드로 인덱싱을 진행하고 있습니다 (약 5분 소요)...';

                // Fallback: 기존 동기식 /api/index 호출
                try {
                    const fallbackRes = await fetch('/api/index', { method: 'POST', body: formData });
                    const fbData = await fallbackRes.json();
                    if (!fallbackRes.ok) {
                        if (fbData && fbData.diagnosis) {
                            showDiagnosisModal(fbData.diagnosis);
                        }
                        throw new Error(fbData.detail || '인덱싱 실패');
                    }

                    if (fbData.skipped) {
                        // 하위 문서 또는 중복 파일 스킵 시 처리
                        if (fbData.step_a1_load) {
                            currentDocData.fileName = fbData.step_a1_load.file_name;
                            currentDocData.totalPages = fbData.step_a1_load.total_pages;
                            currentDocData.totalChars = fbData.step_a1_load.total_chars;
                            currentDocData.pages = fbData.step_a1_load.pages_content || [];
                            currentDocData.all_chunks = [];
                            currentDocData.currentCardPage = 1;
                            currentDocData.currentModalPage = 1;
                            renderA1Card();
                        }
                        setStepIndicatorState('A1', 'done');
                        ['A2', 'A3', 'A4'].forEach(id => {
                            setStepIndicatorState(id, 'waiting');
                            const b = document.getElementById('badge' + id);
                            if (b) {
                                b.className = 'text-[10px] px-2 py-0.5 rounded bg-amber-100 text-amber-800 font-medium';
                                b.innerText = '중복 스킵';
                            }
                        });

                        const alertBanner = document.getElementById('subsetAlertBanner');
                        const alertText = document.getElementById('subsetAlertText');
                        const alertTitle = document.getElementById('subsetAlertTitle');
                        const ratioBadge = document.getElementById('subsetRatioBadge');
                        if (alertBanner && alertText) {
                            alertBanner.classList.remove('hidden');
                            alertText.innerText = fbData.message;
                            if (fbData.subset_info) {
                                const ratioPct = Math.round(fbData.subset_info.overlap_ratio * 100);
                                if (ratioBadge) ratioBadge.innerText = `일치율 ${ratioPct}%`;
                                if (alertTitle) alertTitle.innerText = `'${fbData.subset_info.matched_file}'의 부분 발췌본(하위 문서) 감지`;
                            }
                        }

                        if (pBar) {
                            pBar.className = 'bg-gradient-to-r from-amber-500 via-yellow-400 to-amber-300 h-full rounded-full transition-all duration-300';
                            pBar.style.width = '100%';
                        }
                        if (pText) pText.innerText = '100%';
                        if (pBadge) pBadge.innerText = fbData.skipped_as_subset ? '하위 문서 감지' : '중복 파일 감지';
                        if (pTextBody) pTextBody.innerText = fbData.message;
                        updateStatus();
                        return;
                    }

                    // A1 ~ A4 렌더링
                    currentDocData.fileName = fbData.step_a1_load.file_name;
                    currentDocData.totalPages = fbData.step_a1_load.total_pages;
                    currentDocData.totalChars = fbData.step_a1_load.total_chars;
                    currentDocData.pages = fbData.step_a1_load.pages_content || [];
                    currentDocData.all_chunks = fbData.step_a2_split.chunks || [];
                    currentDocData.currentCardPage = 1;
                    currentDocData.currentModalPage = 1;

                    currentDocData.pages.forEach(p => {
                        p.chunks = currentDocData.all_chunks.filter(c => c.page === p.page);
                        p.chunk_count = p.chunks.length;
                    });

                    renderA1Card();
                    renderA2Card(fbData.step_a2_split);

                    const outA3 = document.getElementById('outputA3');
                    if (outA3) {
                        const sampleVec = fbData.step_a3_embed.sample_vector.slice(0, 3).join(', ');
                        outA3.innerHTML = `
                            <div>임베딩: <b class="text-emerald-700">${fbData.step_a3_embed.vector_count}개</b> (${fbData.step_a3_embed.dimension}차원)</div>
                            <div>모델: gemini-embedding-2</div>
                            <div class="text-[10px] text-slate-400 font-mono truncate">[${sampleVec}, ...]</div>
                        `;
                    }
                    const outA4 = document.getElementById('outputA4');
                    if (outA4) {
                        outA4.innerHTML = `
                            <div>저장 레코드: <b>${fbData.step_a4_store.saved_count}개</b></div>
                            <div>DB 파일: ${fbData.step_a4_store.file_size_kb} KB</div>
                            <div class="text-[10px] text-emerald-600 font-semibold">SQLite BLOB + 출처 매핑 완료</div>
                        `;
                    }

                    ['A1', 'A2', 'A3', 'A4'].forEach(id => {
                        const b = document.getElementById('badge' + id);
                        if (b) {
                            b.className = 'text-[10px] px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 font-medium';
                            b.innerText = '완료';
                        }
                        setStepIndicatorState(id, 'done');
                    });

                    if (pBar) {
                        pBar.style.width = '100%';
                        pBar.className = 'bg-gradient-to-r from-emerald-500 via-teal-400 to-emerald-300 h-full rounded-full transition-all duration-300';
                    }
                    if (pText) pText.innerText = '100%';
                    if (pBadge) pBadge.innerText = '전체 완료';
                    if (pTextBody) pTextBody.innerText = '인덱싱 파이프라인 전체 완료!';
                    updateStatus();
                    alert('인덱싱이 성공적으로 완료되었습니다!');
                } catch (fallbackErr) {
                    console.error('인덱싱 최종 실패:', fallbackErr);
                    alert('인덱싱 실패: ' + fallbackErr.message);
                    ['A1', 'A2', 'A3', 'A4'].forEach(id => {
                        const b = document.getElementById('badge' + id);
                        if (b) {
                            b.className = 'text-[10px] px-2 py-0.5 rounded bg-rose-100 text-rose-800 font-medium';
                            b.innerText = '실패';
                        }
                    });
                    if (pBar) pBar.className = 'bg-rose-500 h-full rounded-full transition-all duration-300';
                    if (pBadge) {
                        pBadge.className = 'px-2.5 py-1 rounded-lg bg-rose-500 text-white font-bold text-xs';
                        pBadge.innerText = '오류';
                    }
                    if (pIcon) pIcon.className = 'fa-solid fa-triangle-exclamation text-rose-400';
                    if (pTextBody) pTextBody.innerText = '인덱싱 중 오류가 발생했습니다.';
                    if (pDetail) pDetail.innerText = fallbackErr.message;
                }
            } finally {
                if (progressTimerInterval) {
                    clearInterval(progressTimerInterval);
                    progressTimerInterval = null;
                }
                if (btn) {
                    btn.disabled = false;
                    btn.innerHTML = '<i class="fa-solid fa-play"></i> 인덱싱 시작';
                }
            }
            return false;
        }

        // B단계: 질문하기 처리
        async function handleAsk(e) {
            e.preventDefault();
            const query = document.getElementById('queryInput').value.trim();
            if (!query) return;

            const btn = document.getElementById('btnAsk');
            btn.disabled = true;
            btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> 검색 및 답변 중...';

            ['B1', 'B2', 'B3', 'B4'].forEach(id => {
                const b = document.getElementById('badge' + id);
                b.className = 'text-[10px] px-2 py-0.5 rounded bg-amber-100 text-amber-800 font-medium';
                b.innerText = '처리 중';
            });

            const topK = parseInt(document.getElementById('topKInput').value) || 3;
            const llmModel = document.getElementById('modelSelect').value;

            try {
                const res = await fetch('/api/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query, top_k: topK, llm_model: llmModel })
                });
                const data = await res.json();
                if (!res.ok) {
                    if (data && data.diagnosis) {
                        showDiagnosisModal(data.diagnosis);
                        throw new Error(data.diagnosis.category_name);
                    }
                    throw new Error(typeof data.detail === 'string' ? data.detail : '답변 생성 실패');
                }

                // B1 출력 업데이트
                const qVec = data.step_b1_query.sample_vector.slice(0, 3).join(', ');
                document.getElementById('outputB1').innerHTML = `
                    <div class="truncate">질문: "${data.step_b1_query.question}"</div>
                    <div>임베딩 차원: <b>${data.step_b1_query.dimension}차원</b></div>
                    <div class="text-[10px] text-slate-400 font-mono truncate">[${qVec}, ...]</div>
                `;
                document.getElementById('badgeB1').className = 'text-[10px] px-2 py-0.5 rounded bg-amber-100 text-amber-800 font-medium';
                document.getElementById('badgeB1').innerText = '완료';
                document.getElementById('timeB1').innerText = `${data.step_b1_query.elapsed}s`;

                // B2 출력 업데이트 (하이브리드 RRF 검색 결과)
                let b2Html = `<div class="flex items-center justify-between pb-0.5 border-b border-slate-100">
                    <span>검색 결과: <b>${data.step_b2_retrieval.retrieved_count}개</b> 청크</span>
                    <span class="text-[10px] bg-amber-50 text-amber-700 px-1.5 py-0.2 rounded font-semibold border border-amber-200">하이브리드 RRF</span>
                </div>`;
                data.step_b2_retrieval.results_preview.forEach(item => {
                    const rankInfo = (item.dense_rank && item.bm25_rank)
                        ? `<span class="text-slate-400 font-normal text-[9.5px]">(V:#${item.dense_rank}, K:#${item.bm25_rank})</span>`
                        : '';
                    b2Html += `<div class="p-1.5 my-1 bg-slate-50 border border-slate-200 rounded text-[10.5px]">
                        <div class="flex items-center justify-between font-bold text-amber-900 mb-0.5">
                            <span class="flex items-center gap-1">
                                <span class="bg-amber-100 text-amber-800 px-1 py-0.2 rounded text-[10px]">#${item.rank}</span>
                                <span>점수: ${item.score}</span>
                                ${rankInfo}
                            </span>
                            <span class="text-[10px] bg-slate-100 text-slate-600 px-1 py-0.2 rounded font-semibold">p.${item.page}</span>
                        </div>
                        <div class="truncate text-slate-600 text-[10px]">${item.preview}</div>
                    </div>`;
                });
                document.getElementById('outputB2').innerHTML = b2Html;
                document.getElementById('badgeB2').className = 'text-[10px] px-2 py-0.5 rounded bg-amber-100 text-amber-800 font-medium';
                document.getElementById('badgeB2').innerText = '완료';
                document.getElementById('timeB2').innerText = `${data.step_b2_retrieval.elapsed}s`;

                // B3 출력 업데이트
                document.getElementById('outputB3').innerHTML = `
                    <div>지시문: ${data.step_b3_prompt.system_instruction_len}자</div>
                    <div>결합 컨텍스트: ${data.step_b3_prompt.context_str_len}자</div>
                    <div class="text-[10px] text-slate-400 truncate">원 질문 보존: "${data.query}"</div>
                `;
                document.getElementById('badgeB3').className = 'text-[10px] px-2 py-0.5 rounded bg-amber-100 text-amber-800 font-medium';
                document.getElementById('badgeB3').innerText = '완료';
                document.getElementById('timeB3').innerText = `${data.step_b3_prompt.elapsed}s`;

                // B4 출력 및 최종 답변 업데이트
                const isGemma = data.step_b4_generation.model.toLowerCase().includes('gemma');
                const modelBadge = isGemma
                    ? `<span class="bg-purple-100 text-purple-800 px-1.5 py-0.5 rounded text-[10px] font-bold">🦙 ${data.step_b4_generation.model} (Local)</span>`
                    : `<span class="bg-blue-100 text-blue-800 px-1.5 py-0.5 rounded text-[10px] font-bold">💎 ${data.step_b4_generation.model} (Cloud)</span>`;

                document.getElementById('outputB4').innerHTML = `
                    <div class="flex items-center gap-1 mb-1">모델: ${modelBadge}</div>
                    <div>답변 길이: <b>${data.step_b4_generation.answer_len}자</b></div>
                    <div class="text-[10px] text-slate-400">생성 시간: ${data.step_b4_generation.latency}초</div>
                `;
                document.getElementById('badgeB4').className = 'text-[10px] px-2 py-0.5 rounded bg-amber-100 text-amber-800 font-medium';
                document.getElementById('badgeB4').innerText = '완료';
                document.getElementById('timeB4').innerText = `${data.step_b4_generation.latency}s`;

                document.getElementById('finalAnswerBox').innerText = data.answer;
                document.getElementById('answerLatency').innerText = `사용 모델: ${data.step_b4_generation.model} · 응답 소요 시간: ${data.step_b4_generation.latency}초`;

            } catch (err) {
                alert('에러 발생: ' + err.message);
                ['B1', 'B2', 'B3', 'B4'].forEach(id => {
                    const b = document.getElementById('badge' + id);
                    b.className = 'text-[10px] px-2 py-0.5 rounded bg-rose-100 text-rose-800 font-medium';
                    b.innerText = '실패';
                });
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<i class="fa-solid fa-magnifying-glass"></i> 질문하기';
            }
        }
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    uvicorn.run("rag_app:app", host="127.0.0.1", port=8000, reload=True, app_dir=str(CURRENT_DIR))

