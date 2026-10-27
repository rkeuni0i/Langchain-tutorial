"""FastAPI 기반 LangChain RAG 웹 애플리케이션

첨부 다이어그램의 단계별(A1~A4, B1~B4) 파이프라인 출력을
웹 대시보드와 REST API(/docs)를 통해 제공합니다.
"""

import os
import shutil
import sys
import tempfile
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
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

try:
    from .rag_pipeline import PDFRAGPipeline, RAGConfig
    from .sqlite_vector_store import SQLiteVectorStore
except ImportError:
    from rag_pipeline import PDFRAGPipeline, RAGConfig
    from sqlite_vector_store import SQLiteVectorStore

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
            "db_path": pipeline.config.db_path,
        },
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


@app.post("/api/index")
async def index_pdf_file(
    file: UploadFile = File(...),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(100),
    clear_existing: bool = Form(True),
):
    """[A단계] PDF 파일 업로드 및 인덱싱 (스텝 A1 ~ A4 실행)"""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드할 수 있습니다.")

    # 사용자 지정 chunk 설정 반영
    pipeline.config.chunk_size = chunk_size
    pipeline.config.chunk_overlap = chunk_overlap

    # 업로드 파일 저장
    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # A단계 인덱싱 파이프라인 실행
        result = pipeline.index_pdf(str(save_path), clear_existing=clear_existing)

        # JSON 직렬화에 적합하게 Document/원시 객체는 제외하고 통계와 미리보기 정보 반환
        return {
            "success": True,
            "file_name": file.filename,
            "config": {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "embedding_model": pipeline.config.embedding_model,
            },
            "step_a1_load": {
                "file_name": result["step1_load"]["file_name"],
                "file_size_kb": result["step1_load"]["file_size_kb"],
                "total_pages": result["step1_load"]["total_pages"],
                "total_chars": result["step1_load"]["total_chars"],
                "sample_preview": result["step1_load"]["sample_preview"],
                "pages_content": result["step1_load"]["pages_content"],
                "elapsed": result["step1_load"]["elapsed"],
            },
            "step_a2_split": {
                "chunk_count": result["step2_split"]["chunk_count"],
                "min_len": result["step2_split"]["min_len"],
                "max_len": result["step2_split"]["max_len"],
                "avg_len": result["step2_split"]["avg_len"],
                "sample_first": result["step2_split"]["sample_first"],
                "sample_last": result["step2_split"]["sample_last"],
                "elapsed": result["step2_split"]["elapsed"],
            },
            "step_a3_embed": {
                "vector_count": result["step3_embed"]["vector_count"],
                "dimension": result["step3_embed"]["dimension"],
                "sample_vector": result["step3_embed"]["sample_vector"],
                "elapsed": result["step3_embed"]["elapsed"],
            },
            "step_a4_store": {
                "saved_count": result["step4_store"]["saved_count"],
                "total_in_db": result["step4_store"]["total_in_db"],
                "db_path": result["step4_store"]["db_path"],
                "file_size_kb": result["step4_store"]["file_size_kb"],
                "elapsed": result["step4_store"]["elapsed"],
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"인덱싱 처리 중 오류 발생: {str(e)}")


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
        result = pipeline.ask(request.query)

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
                    <p class="text-xs text-slate-500">Gemini 임베딩-2 & gemini-3.8-flash & SQLite 벡터 DB 기반</p>
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
            <form id="indexForm" onsubmit="handleIndex(event)" class="grid grid-cols-1 md:grid-cols-12 gap-4 mb-6 bg-emerald-50/60 p-4 rounded-xl border border-emerald-100">
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
                    <button type="submit" id="btnIndex"
                            class="w-full bg-emerald-600 hover:bg-emerald-700 text-white font-medium text-xs py-2.5 px-4 rounded-lg shadow-sm transition flex items-center justify-center gap-1.5">
                        <i class="fa-solid fa-play"></i> 인덱싱 시작
                    </button>
                </div>
            </form>

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
                    <div class="w-32">
                        <label class="block text-xs font-semibold text-slate-700 mb-1">
                            검색 청크 수 (top_k)
                        </label>
                        <input type="number" id="topKInput" value="3" min="1" max="10"
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
                            <div class="text-slate-400 text-center py-5">gemini-3.8-flash 대기</div>
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

            <!-- 모달 본문: 해당 페이지 전체 내용 -->
            <div class="p-6 flex-1 overflow-y-auto bg-slate-100/50">
                <div class="flex items-center justify-between mb-2">
                    <span class="text-xs font-semibold text-slate-500">
                        <i class="fa-solid fa-bookmark text-emerald-600 mr-1"></i>
                        현재 위치: <span id="modalCurrentPageBadge" class="text-emerald-700 font-bold">1 페이지</span>
                    </span>
                    <span id="modalPageCharCount" class="text-xs text-slate-400 font-medium">0자</span>
                </div>
                <div id="modalPageTextBody" 
                     class="w-full bg-white border border-slate-200 rounded-xl p-5 text-slate-800 text-xs sm:text-sm leading-relaxed whitespace-pre-wrap font-mono min-h-[350px] max-h-[60vh] overflow-y-auto shadow-inner select-text">
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

    <script>
        // 전역 문서 페이지 상태
        let currentDocData = {
            fileName: '',
            totalPages: 0,
            totalChars: 0,
            pages: [],
            currentCardPage: 1,
            currentModalPage: 1
        };

        // 초기 상태 확인 및 문서 페이지 로드
        async function updateStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                document.getElementById('chunkCount').innerText = data.total_chunks_in_db;

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
                    currentDocData.currentCardPage = 1;
                    currentDocData.currentModalPage = 1;

                    renderA1Card();
                    document.getElementById('badgeA1').className = 'text-[10px] px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 font-medium';
                    document.getElementById('badgeA1').innerText = '완료';
                }
            } catch (err) {
                console.error('문서 페이지 로드 실패:', err);
            }
        }

        // A1 카드 내용 렌더링 (페이지별 확인 및 미리보기)
        function renderA1Card() {
            if (!currentDocData.pages.length) return;

            const pIdx = Math.max(1, Math.min(currentDocData.currentCardPage, currentDocData.pages.length));
            currentDocData.currentCardPage = pIdx;
            const pageItem = currentDocData.pages[pIdx - 1];

            document.getElementById('outputA1').innerHTML = `
                <div class="flex items-center justify-between pb-1 border-b border-slate-100">
                    <span class="font-bold text-slate-800 truncate" title="${currentDocData.fileName}">${currentDocData.fileName}</span>
                    <span class="text-[10px] bg-emerald-50 text-emerald-700 border border-emerald-100 px-1.5 py-0.5 rounded font-semibold">${currentDocData.totalPages}p (${currentDocData.totalChars.toLocaleString()}자)</span>
                </div>
                <div class="mt-1 flex items-center justify-between text-[11px] text-emerald-900 font-medium bg-emerald-50/80 px-2 py-0.5 rounded">
                    <span><i class="fa-solid fa-file-lines mr-1 text-emerald-600"></i>${pageItem.page} 페이지 내용</span>
                    <span class="text-[10px] text-slate-400 font-normal">${pageItem.char_count.toLocaleString()}자</span>
                </div>
                <div class="text-[10.5px] text-slate-600 bg-slate-50 p-1.5 rounded border border-slate-100 max-h-[75px] overflow-y-auto whitespace-pre-wrap font-mono leading-relaxed select-text mt-1">
${pageItem.text}
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

        // 모달 열기
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
            document.getElementById('modalPageCharCount').innerText = `글자 수: ${pageItem.char_count.toLocaleString()}자`;
            document.getElementById('modalPageTextBody').innerText = pageItem.text;

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
            const modal = document.getElementById('pageViewerModal');
            if (!modal || modal.classList.contains('hidden')) return;

            if (e.key === 'ArrowLeft') {
                e.preventDefault();
                navigatePage(-1);
            } else if (e.key === 'ArrowRight') {
                e.preventDefault();
                navigatePage(1);
            } else if (e.key === 'Escape') {
                e.preventDefault();
                closePageModal();
            }
        });

        // DB 초기화
        async function clearDb() {
            if (!confirm('정말로 SQLite 벡터 DB를 초기화하시겠습니까?')) return;
            try {
                const res = await fetch('/api/clear', { method: 'DELETE' });
                const data = await res.json();
                alert(data.message);
                currentDocData = { fileName: '', totalPages: 0, totalChars: 0, pages: [], currentCardPage: 1, currentModalPage: 1 };
                document.getElementById('outputA1').innerHTML = '<div class="text-slate-400 text-center py-5">PDF 문서를 업로드하세요</div>';
                const nav = document.getElementById('pageNavMiniA1');
                if (nav) {
                    nav.classList.add('hidden');
                    nav.classList.remove('flex');
                }
                document.getElementById('badgeA1').className = 'text-[10px] px-2 py-0.5 rounded bg-slate-200 text-slate-600 font-medium';
                document.getElementById('badgeA1').innerText = '대기';
                updateStatus();
            } catch (e) {
                alert('초기화 실패: ' + e);
            }
        }

        // A단계: 인덱싱 제출 처리
        async function handleIndex(e) {
            e.preventDefault();
            const fileInput = document.getElementById('pdfFileInput');
            if (!fileInput.files.length) {
                alert('PDF 파일을 선택해 주세요.');
                return;
            }

            const btn = document.getElementById('btnIndex');
            btn.disabled = true;
            btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> 인덱싱 중...';

            // 뱃지 상태 업데이트
            ['A1', 'A2', 'A3', 'A4'].forEach(id => {
                const b = document.getElementById('badge' + id);
                b.className = 'text-[10px] px-2 py-0.5 rounded bg-amber-100 text-amber-800 font-medium';
                b.innerText = '처리 중';
            });

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            formData.append('chunk_size', document.getElementById('chunkSizeInput').value);
            formData.append('chunk_overlap', document.getElementById('chunkOverlapInput').value);
            formData.append('clear_existing', 'true');

            try {
                const res = await fetch('/api/index', { method: 'POST', body: formData });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || '인덱싱 실패');

                // A1 출력 및 데이터 업데이트
                currentDocData.fileName = data.step_a1_load.file_name;
                currentDocData.totalPages = data.step_a1_load.total_pages;
                currentDocData.totalChars = data.step_a1_load.total_chars;
                currentDocData.pages = data.step_a1_load.pages_content || [];
                currentDocData.currentCardPage = 1;
                currentDocData.currentModalPage = 1;

                renderA1Card();
                document.getElementById('badgeA1').className = 'text-[10px] px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 font-medium';
                document.getElementById('badgeA1').innerText = '완료';
                document.getElementById('timeA1').innerText = `${data.step_a1_load.elapsed}s`;

                // A2 출력 업데이트
                document.getElementById('outputA2').innerHTML = `
                    <div>분할 청크: <b class="text-emerald-700">${data.step_a2_split.chunk_count}개</b></div>
                    <div>길이: 최소 ${data.step_a2_split.min_len} / 최대 ${data.step_a2_split.max_len} / 평균 ${data.step_a2_split.avg_len}자</div>
                    <div class="text-[10px] text-slate-400 truncate">#1: "${data.step_a2_split.sample_first}"</div>
                `;
                document.getElementById('badgeA2').className = 'text-[10px] px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 font-medium';
                document.getElementById('badgeA2').innerText = '완료';
                document.getElementById('timeA2').innerText = `${data.step_a2_split.elapsed}s`;

                // A3 출력 업데이트
                const sampleVec = data.step_a3_embed.sample_vector.slice(0, 3).join(', ');
                document.getElementById('outputA3').innerHTML = `
                    <div>임베딩: <b class="text-emerald-700">${data.step_a3_embed.vector_count}개</b> (${data.step_a3_embed.dimension}차원)</div>
                    <div>모델: gemini-embedding-2</div>
                    <div class="text-[10px] text-slate-400 font-mono truncate">[${sampleVec}, ...]</div>
                `;
                document.getElementById('badgeA3').className = 'text-[10px] px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 font-medium';
                document.getElementById('badgeA3').innerText = '완료';
                document.getElementById('timeA3').innerText = `${data.step_a3_embed.elapsed}s`;

                // A4 출력 업데이트
                document.getElementById('outputA4').innerHTML = `
                    <div>저장 레코드: <b>${data.step_a4_store.saved_count}개</b></div>
                    <div>DB 파일: ${data.step_a4_store.file_size_kb} KB</div>
                    <div class="text-[10px] text-emerald-600 font-semibold">SQLite BLOB + 출처 매핑 완료</div>
                `;
                document.getElementById('badgeA4').className = 'text-[10px] px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 font-medium';
                document.getElementById('badgeA4').innerText = '완료';
                document.getElementById('timeA4').innerText = `${data.step_a4_store.elapsed}s`;

                updateStatus();
                alert('A단계 (문서 읽기 → 청크 분할 → 임베딩 → SQLite 저장) 인덱싱이 완료되었습니다!');
            } catch (err) {
                alert('에러 발생: ' + err.message);
                ['A1', 'A2', 'A3', 'A4'].forEach(id => {
                    const b = document.getElementById('badge' + id);
                    b.className = 'text-[10px] px-2 py-0.5 rounded bg-rose-100 text-rose-800 font-medium';
                    b.innerText = '실패';
                });
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<i class="fa-solid fa-play"></i> 인덱싱 시작';
            }
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

            try {
                const res = await fetch('/api/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query, top_k: topK })
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || '답변 생성 실패');

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

                // B2 출력 업데이트
                let b2Html = `<div>검색 결과: <b>${data.step_b2_retrieval.retrieved_count}개</b> 청크</div>`;
                data.step_b2_retrieval.results_preview.forEach(item => {
                    b2Html += `<div class="p-1 my-1 bg-slate-50 border rounded text-[10px]">
                        <span class="text-amber-800 font-bold">#${item.rank} [유사도: ${item.score}]</span> (p.${item.page})
                        <div class="truncate text-slate-500">${item.preview}</div>
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
                document.getElementById('outputB4').innerHTML = `
                    <div>모델: <b>${data.step_b4_generation.model}</b></div>
                    <div>답변 길이: ${data.step_b4_generation.answer_len}자</div>
                `;
                document.getElementById('badgeB4').className = 'text-[10px] px-2 py-0.5 rounded bg-amber-100 text-amber-800 font-medium';
                document.getElementById('badgeB4').innerText = '완료';
                document.getElementById('timeB4').innerText = `${data.step_b4_generation.latency}s`;

                document.getElementById('finalAnswerBox').innerText = data.answer;
                document.getElementById('answerLatency').innerText = `응답 소요 시간: ${data.step_b4_generation.latency}초`;

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

