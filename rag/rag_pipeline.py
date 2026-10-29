"""PDF 기반 LangChain RAG 파이프라인

다이어그램 구조(A. 미리 준비하기 / B. 질문에 답하기)의 
각 스텝별 출력값을 콘솔에서 투명하게 확인할 수 있도록 구현되었습니다.
"""

import hashlib
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

# 현재 디렉토리 sys.path 보정
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import google.genai as genai
from google.genai import types
import pypdf
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from .sqlite_vector_store import SQLiteVectorStore
except ImportError:
    from sqlite_vector_store import SQLiteVectorStore

try:
    from .gemini_error_debugger import diagnose_gemini_error, print_gemini_429_diagnosis
except ImportError:
    from gemini_error_debugger import diagnose_gemini_error, print_gemini_429_diagnosis


DEFAULT_OCR_PROMPT = (
    "다음 제공된 PDF 문서에 포함된 모든 글자(텍스트)를 누락 없이 있는 그대로 OCR 추출해 주세요.\n\n"
    "[엄격한 준수 사항]\n"
    "1. 사진, 일러스트, 그래프, 다이어그램 등 이미지에 대한 설명이나 시각적 묘사는 모두 빼고 완전히 제외하세요.\n"
    "2. 오직 문서에 적혀 있는 순수한 문자/텍스트(활자) 내용만을 그대로 추출하세요.\n"
    "3. 각 페이지의 시작 부분에 반드시 '--- [PAGE N] ---' (예: --- [PAGE 1] ---) 형식의 페이지 구분자를 넣어주세요.\n"
    "4. 부가적인 인사말, 머리말, 마크다운 코드 블록(```) 등의 래핑 없이 오직 추출된 텍스트만 출력하세요."
)


@dataclass
class RAGConfig:
    """사용자가 변경 가능한 RAG 설정값 클래스"""

    # A. 청크 분할 설정
    chunk_size: int = 800  # 청크 분할 크기 (사용자 변경 가능)
    chunk_overlap: int = 150  # 청크 간 중복 글자 수 (사용자 변경 가능)

    # B. 벡터 스토어 및 검색 설정
    db_path: str = str(CURRENT_DIR / "rag_sqlite.db")  # SQLite 파일 경로 (절대경로)
    top_k: int = 4  # 검색할 관련 청크 수 (기본 4개)

    # C. 모델 설정
    embedding_model: str = "gemini-embedding-2"  # Gemini 2세대 임베딩 모델
    llm_model: str = "gemini-3.8-flash"  # Gemini 또는 Gemma LLM 모델 ("gemini-3.8-flash" | "gemma4:e2b")
    ollama_base_url: str = "http://127.0.0.1:11434"  # Ollama 로컬 서버 API 주소
    temperature: float = 0.2  # LLM 응답 창의성 (0.0: 보수적/정확, 1.0: 창의적)

    # D. OCR 프롬프트 설정 (사용자 직접 수정 가능)
    ocr_prompt: str = DEFAULT_OCR_PROMPT


class PDFRAGPipeline:
    def __init__(self, config: Optional[RAGConfig] = None):
        # 환경변수 로드 (.env 탐색)
        for env_file in [PROJECT_ROOT / ".env", CURRENT_DIR / ".env"]:
            if env_file.exists():
                load_dotenv(dotenv_path=env_file, override=True)
        load_dotenv(override=True)
        self.config = config or RAGConfig()

        # 모델명 별칭 정규화 (예: gemini-embedding-002 -> gemini-embedding-2)
        model_name = self.config.embedding_model
        if "gemini-embedding-002" in model_name:
            model_name = model_name.replace("gemini-embedding-002", "gemini-embedding-2")
            self.config.embedding_model = model_name

        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY 또는 GOOGLE_API_KEY 환경변수가 설정되지 않았습니다. .env 파일을 확인해 주세요."
            )

        # 1. Google GenAI 클라이언트 (멀티모달 PDF OCR용)
        self.genai_client = genai.Client(api_key=self.api_key)

        # 2. 임베딩 모델 초기화
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=self.config.embedding_model,
            google_api_key=self.api_key,
        )

        # 3. LLM 모델 초기화 (Gemini 또는 Ollama)
        self.llm = self._create_llm(self.config.llm_model, self.config.temperature)

        # 4. SQLite 벡터 스토어 초기화
        self.vector_store = SQLiteVectorStore(db_path=self.config.db_path)

        # 5. 최근 인덱싱된 문서 정보 캐시 (페이지별 전체 텍스트 뷰어용)
        self.cache_path = os.path.join(
            os.path.dirname(os.path.abspath(self.config.db_path)),
            "uploads",
            "last_indexed_info.json",
        )
        self.last_indexed_info = {
            "file_name": "",
            "total_pages": 0,
            "total_chars": 0,
            "pages_content": [],
            "step_a1": None,
        }
        self._load_cache()

    def _load_cache(self) -> None:
        """이전 인덱싱 캐시 파일 로드"""
        if os.path.exists(self.cache_path) and self.vector_store.count() > 0:
            try:
                import json

                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.last_indexed_info = json.load(f)
            except Exception:
                pass

    def _save_cache(self) -> None:
        """인덱싱된 문서 정보를 JSON 파일로 캐시 저장"""
        try:
            import json

            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.last_indexed_info, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _create_llm(self, model_name: str, temperature: Optional[float] = None) -> Any:
        """모델명(Gemini vs Gemma/Ollama)에 따라 적절한 LLM 객체 동적 생성"""
        temp = temperature if temperature is not None else self.config.temperature
        model_lower = model_name.lower()
        if "gemma" in model_lower or "ollama" in model_lower:
            if ChatOllama is None:
                raise ImportError(
                    "Ollama 모델을 사용하려면 langchain-ollama 패키지가 필요합니다."
                )
            return ChatOllama(
                model=model_name,
                base_url=self.config.ollama_base_url,
                temperature=temp,
            )
        else:
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=self.api_key,
                temperature=temp,
                timeout=30,
                max_retries=3,
                streaming=False,
            )

    def clear_cache(self) -> None:
        """캐시 정보 및 캐시 파일 초기화"""
        self.last_indexed_info = {
            "file_name": "",
            "total_pages": 0,
            "total_chars": 0,
            "pages_content": [],
            "step_a1": None,
        }
        if os.path.exists(self.cache_path):
            try:
                os.remove(self.cache_path)
            except Exception:
                pass

    def get_document_pages(self) -> Dict[str, Any]:
        """인덱싱된 문서의 페이지별 원문 및 분할 청크 세부내역 조회"""
        try:
            if self.vector_store.count() == 0:
                return {"available": False, "pages": [], "all_chunks": []}

            with self.vector_store._get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT DISTINCT source_file FROM rag_chunks LIMIT 1;")
                row = cur.fetchone()
                file_name = row["source_file"] if row else "PDF 문서"

                cur.execute(
                    "SELECT id, page_number, chunk_index, content FROM rag_chunks ORDER BY chunk_index;"
                )
                rows = cur.fetchall()

                # 전체 청크 목록 및 페이지별 청크 그룹화
                all_chunks = []
                chunks_by_page: Dict[int, List[Dict[str, Any]]] = {}
                for r in rows:
                    c_item = {
                        "chunk_index": r["chunk_index"],
                        "page": r["page_number"],
                        "content": r["content"],
                        "char_count": len(r["content"]),
                    }
                    all_chunks.append(c_item)
                    p = r["page_number"]
                    if p not in chunks_by_page:
                        chunks_by_page[p] = []
                    chunks_by_page[p].append(c_item)

                # 페이지별 원문 캐시 확인
                cached_pages = {}
                if self.last_indexed_info and self.last_indexed_info.get("pages_content"):
                    for cp in self.last_indexed_info["pages_content"]:
                        cached_pages[cp["page"]] = cp["text"]

                pages = []
                total_chars = 0
                for p in sorted(chunks_by_page.keys()):
                    p_chunks = chunks_by_page[p]
                    p_text = cached_pages.get(p) or "\n\n".join(c["content"] for c in p_chunks)
                    total_chars += len(p_text)
                    pages.append(
                        {
                            "page": p,
                            "text": p_text,
                            "char_count": len(p_text),
                            "chunk_count": len(p_chunks),
                            "chunks": p_chunks,
                        }
                    )

                return {
                    "available": True,
                    "file_name": file_name,
                    "total_pages": len(pages),
                    "total_chars": total_chars,
                    "total_chunks": len(all_chunks),
                    "pages": pages,
                    "all_chunks": all_chunks,
                }
        except Exception as e:
            return {"available": False, "pages": [], "all_chunks": [], "error": str(e)}

    # =========================================================================
    # 페이지 단위 증분 인덱싱 헬퍼 (이미 적재된 쪽은 OCR/임베딩/저장을 건너뜀)
    # =========================================================================

    @staticmethod
    def _page_fingerprint(page: Any) -> str:
        """PDF 페이지 원본(콘텐츠 스트림 + 삽입 이미지)의 SHA-256. 파일이 달라도 같은 쪽이면 같은 값."""
        h = hashlib.sha256()
        contents = page.get_contents()
        if contents is not None:
            h.update(contents.get_data())
        try:
            for img in page.images:
                h.update(img.data)
        except Exception:
            h.update(b"no-image-data")
        return h.hexdigest()

    def _compute_page_hashes(self, pdf_path: str) -> List[str]:
        reader = pypdf.PdfReader(pdf_path)
        return [self._page_fingerprint(pg) for pg in reader.pages]

    def _backfill_page_hashes(self, directory: str) -> None:
        """페이지 해시 기록이 없는 기존 적재 파일(구버전 DB)을 같은 폴더의 원본 PDF로 보강 기록"""
        with self.vector_store._get_connection() as conn:
            files = [r["file_name"] for r in conn.execute("SELECT file_name FROM indexed_files;").fetchall()]
        for fname in files:
            fpath = os.path.join(directory, fname)
            if self.vector_store.has_recorded_pages(fname) or not os.path.exists(fpath):
                continue
            try:
                hashes = self._compute_page_hashes(fpath)
                with self.vector_store._get_connection() as conn:
                    pages_with_chunks = {
                        r["page_number"]
                        for r in conn.execute(
                            "SELECT DISTINCT page_number FROM rag_chunks WHERE source_file = ?;", (fname,)
                        ).fetchall()
                    }
                self.vector_store.record_pages(
                    fname, [(i + 1, h) for i, h in enumerate(hashes) if (i + 1) in pages_with_chunks]
                )
                print(f"  ℹ [페이지 해시 보강] '{fname}' 기존 적재 페이지 {len(pages_with_chunks)}개 기록")
            except Exception as e:
                print(f"  [경고] 페이지 해시 보강 실패({fname}): {e}")

    @staticmethod
    def _write_subset_pdf(pdf_path: str, page_indices: List[int], out_path: str) -> None:
        reader = pypdf.PdfReader(pdf_path)
        writer = pypdf.PdfWriter()
        for i in page_indices:
            writer.add_page(reader.pages[i])
        with open(out_path, "wb") as f:
            writer.write(f)

    # =========================================================================
    # A. 미리 준비하기 (인덱싱: 자료를 등록하거나 변경할 때)
    # =========================================================================

    def index_pdf(
        self,
        pdf_path: str,
        clear_existing: bool = True,
        ocr_prompt: Optional[str] = None,
        progress_callback: Optional[Callable[[int, str, str, Optional[Dict[str, Any]]], None]] = None,
        detect_subset: bool = True,
        force_reindex: bool = False,
    ) -> Dict[str, Any]:
        """A 단계 전체 실행: PDF 읽기 -> 중복/하위 문서 검사 -> 청크 분할 -> 임베딩 -> SQLite 저장"""
        self._print_header("A. 미리 준비하기 (자료를 등록하거나 변경할 때 · 인덱싱)")

        def emit_progress(pct: int, step: str, msg: str, data: Optional[Dict[str, Any]] = None):
            if progress_callback:
                try:
                    progress_callback(pct, step, msg, data)
                except Exception as cb_err:
                    print(f"  [경고] progress_callback 실행 오류: {cb_err}")

        emit_progress(2, "A0", "인덱싱 준비 및 환경 초기화 중...")

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

        file_name = os.path.basename(pdf_path)
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        file_hash = hashlib.sha256(pdf_bytes).hexdigest()
        file_size_kb = round(len(pdf_bytes) / 1024, 1)

        # ---------------------------------------------------------
        # [사전 검사 1] 완전 동일 파일(Exact Duplicate) 감지
        # ---------------------------------------------------------
        if not force_reindex and self.vector_store.count() > 0:
            existing_file = self.vector_store.get_file_by_hash(file_hash)
            if existing_file:
                msg = (
                    f"'{file_name}' 파일은 이미 완전히 동일한 내용(SHA-256 해시 일치)으로 인덱싱되어 있습니다. "
                    f"(등록일시: {existing_file.get('indexed_at')}, {existing_file.get('total_chunks')}개 청크 보존됨) "
                    f"불필요한 중복 저장을 방지하기 위해 신규 인덱싱을 건너뛰었습니다."
                )
                print(f"  ℹ [중복 파일 감지] {msg}")
                emit_progress(100, "DUPLICATE_SKIPPED", msg, {
                    "skipped": True,
                    "skipped_as_duplicate": True,
                    "skip_reason": "exact_duplicate",
                    "file_name": file_name,
                    "existing_file": existing_file,
                })
                return {
                    "success": True,
                    "skipped": True,
                    "skipped_as_duplicate": True,
                    "skip_reason": "exact_duplicate",
                    "message": msg,
                    "file_name": file_name,
                    "existing_file": existing_file,
                }

        # ---------------------------------------------------------
        # [사전 검사 1-2] 페이지 단위 증분 계획: 이미 적재된 쪽(원본 해시 일치)은 건너뛰고 새 쪽만 처리
        # ---------------------------------------------------------
        page_hashes: List[str] = []
        try:
            page_hashes = self._compute_page_hashes(pdf_path)
        except Exception as e:
            print(f"  [경고] 페이지 해시 계산 실패, 전체 처리로 진행합니다: {e}")

        incremental = (
            bool(page_hashes) and not force_reindex and not clear_existing and self.vector_store.count() > 0
        )
        new_indices: List[int] = list(range(len(page_hashes)))
        partial_incremental = False
        if incremental:
            self._backfill_page_hashes(os.path.dirname(os.path.abspath(pdf_path)))
            known = self.vector_store.get_known_page_hashes(page_hashes)
            seen_in_file: set = set()
            new_indices = []
            for i, h in enumerate(page_hashes):
                if h in known or h in seen_in_file:
                    continue
                seen_in_file.add(h)
                new_indices.append(i)

            skipped_pages = len(page_hashes) - len(new_indices)
            if not new_indices:
                msg = (
                    f"'{file_name}'의 {len(page_hashes)}개 페이지는 모두 이미 DB에 적재된 내용입니다. "
                    f"OCR·임베딩·저장을 모두 건너뛰었습니다."
                )
                print(f"  ℹ [전체 페이지 중복] {msg}")
                emit_progress(100, "DUPLICATE_SKIPPED", msg, {
                    "skipped": True, "skipped_as_duplicate": True, "skip_reason": "all_pages_indexed",
                    "file_name": file_name,
                })
                return {
                    "success": True,
                    "skipped": True,
                    "skipped_as_duplicate": True,
                    "skip_reason": "all_pages_indexed",
                    "message": msg,
                    "file_name": file_name,
                    "page_plan": {"total_pages": len(page_hashes), "skipped_pages": skipped_pages, "new_pages": 0},
                }
            partial_incremental = skipped_pages > 0
            if partial_incremental:
                print(
                    f"  ℹ [증분 인덱싱] 전체 {len(page_hashes)}쪽 중 {skipped_pages}쪽은 이미 적재되어 건너뛰고 "
                    f"새 {len(new_indices)}쪽만 처리합니다 (p.{new_indices[0] + 1}~p.{new_indices[-1] + 1})."
                )
                emit_progress(4, "A0", f"이미 적재된 {skipped_pages}쪽은 건너뛰고 새 {len(new_indices)}쪽만 OCR합니다.")

        # ---------------------------------------------------------
        # [스텝 A1] 문서 읽기 (PDF -> 텍스트)
        # ---------------------------------------------------------
        emit_progress(5, "A1", "PDF 문서 로드 및 Gemini 3.8 Flash 멀티모달 OCR 분석 준비 중...")
        if partial_incremental:
            import tempfile

            tmp_dir = tempfile.mkdtemp(prefix="rag_incr_")
            sub_path = os.path.join(tmp_dir, file_name)
            self._write_subset_pdf(pdf_path, new_indices, sub_path)
            try:
                step1_output = self._step_a1_load_pdf(
                    sub_path, ocr_prompt=ocr_prompt, progress_callback=progress_callback
                )
            finally:
                try:
                    os.remove(sub_path)
                    os.rmdir(tmp_dir)
                except OSError:
                    pass
            # 부분 PDF 기준 쪽 번호(1..k)를 원본 PDF 쪽 번호로 환원
            for d in step1_output["documents"]:
                local = int(d.metadata.get("page", 1))
                d.metadata["page"] = new_indices[local - 1] + 1 if 1 <= local <= len(new_indices) else local
                d.metadata["source"] = file_name
                d.metadata["file_path"] = os.path.abspath(pdf_path)
                d.metadata["total_pages"] = len(page_hashes)
            step1_output["pages_content"] = [
                {"page": d.metadata["page"], "text": d.page_content, "char_count": len(d.page_content)}
                for d in step1_output["documents"]
            ]
            step1_output["file_name"] = file_name
            step1_output["total_pages"] = len(page_hashes)
        else:
            step1_output = self._step_a1_load_pdf(
                pdf_path, ocr_prompt=ocr_prompt, progress_callback=progress_callback
            )

        a1_summary = {
            "file_name": step1_output["file_name"],
            "file_size_kb": step1_output["file_size_kb"],
            "total_pages": step1_output["total_pages"],
            "total_chars": step1_output["total_chars"],
            "sample_preview": step1_output["sample_preview"],
            "pages_content": step1_output["pages_content"],
            "elapsed": step1_output["elapsed"],
        }
        emit_progress(
            72,
            "A1",
            f"문서 읽기 완료: 총 {step1_output['total_pages']}페이지 ({step1_output['total_chars']:,}자 추출)",
            a1_summary,
        )

        # ---------------------------------------------------------
        # [사전 검사 2] 하위 문서 / 부분 발췌본(Subset Document) 감지
        # ---------------------------------------------------------
        if not force_reindex and detect_subset and not partial_incremental and self.vector_store.count() > 0:
            full_text = " ".join(d.page_content for d in step1_output["documents"])
            subset_res = self.vector_store.check_subset_document(full_text, threshold=0.75)

            if subset_res.get("is_subset"):
                msg = (
                    f"업로드된 '{file_name}' 문서는 기존 등록된 '{subset_res['matched_file']}'의 "
                    f"내용과 {subset_res['overlap_ratio'] * 100:.1f}% 일치하는 부분 발췌본(하위 문서)입니다. "
                    f"불필요한 임베딩 생성 비용 및 중복 저장을 방지하기 위해 신규 인덱싱을 건너뛰었습니다."
                )
                print(f"  ⚠️ [하위 문서 감지] {msg}")
                emit_progress(100, "SUBSET_SKIPPED", msg, {
                    "skipped": True,
                    "skipped_as_subset": True,
                    "skip_reason": "subset_document",
                    "subset_info": subset_res,
                    "file_name": file_name,
                    "step1_load": step1_output,
                })
                return {
                    "success": True,
                    "skipped": True,
                    "skipped_as_subset": True,
                    "skip_reason": "subset_document",
                    "message": msg,
                    "subset_info": subset_res,
                    "step1_load": step1_output,
                    "step2_split": {
                        "chunk_count": 0,
                        "min_len": 0,
                        "max_len": 0,
                        "avg_len": 0,
                        "sample_first": "(하위 문서 감지로 스킵됨)",
                        "sample_last": "(하위 문서 감지로 스킵됨)",
                        "elapsed": 0.0,
                        "chunk_details": [],
                    },
                    "step3_embed": {
                        "vector_count": 0,
                        "dimension": 0,
                        "sample_vector": [],
                        "elapsed": 0.0,
                    },
                    "step4_store": {
                        "saved_count": 0,
                        "total_in_db": self.vector_store.count(),
                        "db_path": self.config.db_path,
                        "file_size_kb": round(os.path.getsize(self.config.db_path) / 1024, 1) if os.path.exists(self.config.db_path) else 0,
                        "elapsed": 0.0,
                    },
                }

        # 하위 문서가 아닌 신규 문서이거나 강제 재인덱싱인 경우
        if clear_existing:
            self.vector_store.clear()
            self.clear_cache()
            print("  ℹ 기존 SQLite 벡터 스토어 데이터를 초기화했습니다.")

        # ---------------------------------------------------------
        # [스텝 A2] 청크로 나누기 (작은 원문 조각)
        # ---------------------------------------------------------
        emit_progress(
            75,
            "A2",
            f"청크 분할 시작 (크기: {self.config.chunk_size}자, 중복: {self.config.chunk_overlap}자)...",
        )
        step2_output = self._step_a2_split_chunks(step1_output["documents"])

        a2_summary = {
            "chunk_count": step2_output["chunk_count"],
            "min_len": step2_output["min_len"],
            "max_len": step2_output["max_len"],
            "avg_len": step2_output["avg_len"],
            "sample_first": step2_output["sample_first"],
            "sample_last": step2_output["sample_last"],
            "elapsed": step2_output["elapsed"],
            "chunks": step2_output["chunk_details"],
        }
        emit_progress(
            82,
            "A2",
            f"청크 분할 완료: 총 {step2_output['chunk_count']}개 청크 생성 (평균 {step2_output['avg_len']}자)",
            a2_summary,
        )

        # ---------------------------------------------------------
        # [스텝 A3] 임베딩하기 (청크 -> 숫자 배열)
        # ---------------------------------------------------------
        emit_progress(
            85,
            "A3",
            f"{len(step2_output['chunks'])}개 청크 gemini-embedding-2 벡터 생성 중...",
        )
        step3_output = self._step_a3_embed_chunks(step2_output["chunks"])

        a3_summary = {
            "vector_count": step3_output["vector_count"],
            "dimension": step3_output["dimension"],
            "sample_vector": step3_output["sample_vector"],
            "elapsed": step3_output["elapsed"],
            "cache_hit_count": step3_output.get("cache_hit_count", 0),
            "cache_miss_count": step3_output.get("cache_miss_count", len(step3_output["embeddings"])),
        }
        hit_txt = f" (캐시 적중: {a3_summary['cache_hit_count']}개 재사용 ⚡)" if a3_summary['cache_hit_count'] > 0 else ""
        emit_progress(
            94,
            "A3",
            f"임베딩 완료: {step3_output['vector_count']}개 벡터{hit_txt} ({step3_output['dimension']}차원)",
            a3_summary,
        )

        # ---------------------------------------------------------
        # [스텝 A4] 벡터 스토어 저장 (SQLite DB: 벡터 + 원문 + 출처 연결)
        # ---------------------------------------------------------
        emit_progress(96, "A4", "SQLite 벡터 스토어에 벡터 및 메타데이터 저장 중...")
        step4_output = self._step_a4_save_to_sqlite(
            step2_output["chunks"], step3_output["embeddings"]
        )

        # 파일 등록 기록
        self.vector_store.record_file(
            file_name=step1_output["file_name"],
            file_hash=file_hash,
            file_size_kb=file_size_kb,
            total_pages=step1_output["total_pages"],
            total_chunks=step4_output["saved_count"],
        )

        # 적재 완료된 페이지의 원본 해시 기록 (이후 같은 쪽이 다른 PDF로 들어와도 건너뛰기 위함)
        if page_hashes:
            ocr_pages = {int(d.metadata.get("page", 0)) for d in step1_output["documents"]}
            self.vector_store.record_pages(
                file_name,
                [(i + 1, page_hashes[i]) for i in new_indices if (i + 1) in ocr_pages],
            )

        a4_summary = {
            "saved_count": step4_output["saved_count"],
            "total_in_db": step4_output["total_in_db"],
            "db_path": step4_output["db_path"],
            "file_size_kb": step4_output["file_size_kb"],
            "elapsed": step4_output["elapsed"],
        }
        emit_progress(
            100,
            "A4",
            f"인덱싱 완료! 총 {step4_output['saved_count']}개 레코드 SQLite DB 저장 완료",
            a4_summary,
        )

        # 최근 인덱싱된 문서 정보 캐시 업데이트 및 저장
        merged_pages = step1_output["pages_content"]
        if partial_incremental:
            new_nums = {pc["page"] for pc in merged_pages}
            prev = [
                pc for pc in (self.last_indexed_info.get("pages_content") or []) if pc.get("page") not in new_nums
            ]
            merged_pages = sorted(prev + merged_pages, key=lambda pc: pc["page"])
        self.last_indexed_info = {
            "file_name": step1_output["file_name"],
            "total_pages": len(merged_pages) if partial_incremental else step1_output["total_pages"],
            "total_chars": (
                sum(pc.get("char_count", len(pc.get("text", ""))) for pc in merged_pages)
                if partial_incremental
                else step1_output["total_chars"]
            ),
            "pages_content": merged_pages,
            "elapsed": step1_output["elapsed"],
        }
        self._save_cache()

        return {
            "step1_load": step1_output,
            "step2_split": step2_output,
            "step3_embed": step3_output,
            "step4_store": step4_output,
        }

    def _step_a1_load_pdf(
        self,
        pdf_path: str,
        ocr_prompt: Optional[str] = None,
        progress_callback: Optional[Callable[[int, str, str, Optional[Dict[str, Any]]], None]] = None,
    ) -> Dict[str, Any]:
        """스텝 A1. 문서 읽기 (Gemini 3.8 Flash 멀티모달 OCR 추출)
        이미지의 시각적 묘사는 제외하고 순수 텍스트(OCR)만 추출합니다.
        """
        self._print_step("스텝 A1", "문서 읽기 (Gemini 3.8 Flash 멀티모달 OCR)")
        start_time = time.time()

        file_size_kb = os.path.getsize(pdf_path) / 1024
        file_name = os.path.basename(pdf_path)

        # 1. pypdf로 원본 페이지 수 확인
        pypdf_pages = 1
        try:
            reader = pypdf.PdfReader(pdf_path)
            pypdf_pages = len(reader.pages)
        except Exception:
            pass

        # 2. PDF 바이너리 로드
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        # 3. Gemini 3.8 Flash에 전달할 순수 텍스트 OCR 전용 프롬프트
        prompt_to_use = (
            ocr_prompt.strip()
            if (ocr_prompt and ocr_prompt.strip())
            else self.config.ocr_prompt
        )

        print(f"  [진행] Gemini 멀티모달 비전을 통한 PDF 텍스트 OCR 추출 중 (프롬프트 길이: {len(prompt_to_use)}자)...")

        # 장시간 OCR 처리 중 하트비트 주기적 알림 스레드 가동
        stop_heartbeat = threading.Event()

        def heartbeat_worker():
            h_start = time.time()
            while not stop_heartbeat.wait(2.0):
                h_elapsed = time.time() - h_start
                # 5%에서 시작하여 경과 시간에 따라 70%까지 점진적 상승
                prog = min(70, int(5 + (h_elapsed / 320.0) * 65))
                m, s = divmod(int(h_elapsed), 60)
                time_str = f"{m:02d}:{s:02d}"

                if h_elapsed < 15:
                    status_text = f"Gemini 3.8 Flash 멀티모달 모델에 PDF 전달 및 분석 시작... ({time_str} 경과)"
                elif h_elapsed < 60:
                    status_text = f"문서 전체 페이지 비전 판독 및 활자(텍스트) 인식 중... ({time_str} 경과)"
                elif h_elapsed < 180:
                    status_text = f"이미지/그림 제외 및 페이지별 구분자(PAGE N) 정밀 파싱 중... ({time_str} 경과)"
                else:
                    status_text = f"방대한 대용량 문서 활자 텍스트 생성 마무리 단계... ({time_str} 경과)"

                if progress_callback:
                    try:
                        progress_callback(prog, "A1", status_text, None)
                    except Exception:
                        pass

        hb_thread = threading.Thread(target=heartbeat_worker, daemon=True)
        hb_thread.start()

        try:
            response = self.genai_client.models.generate_content(
                model="gemini-3.8-flash",
                contents=[
                    types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                    prompt_to_use,
                ],
            )
        except Exception as e:
            stop_heartbeat.set()
            diag = diagnose_gemini_error(e)
            if diag.get("is_429"):
                print_gemini_429_diagnosis(e)
            raise
        finally:
            stop_heartbeat.set()

        elapsed = time.time() - start_time

        extracted_text = response.text or ""

        # 4. '--- [PAGE N] ---' 패턴으로 페이지별 텍스트 파싱
        page_pattern = re.compile(r"---\s*\[PAGE\s*(\d+)\]\s*---", re.IGNORECASE)
        splits = page_pattern.split(extracted_text)

        documents: List[Document] = []
        if len(splits) > 1:
            # splits = [prefix, page_num_1, text_1, page_num_2, text_2, ...]
            # 앞부분(prefix)에 내용이 있을 경우
            if splits[0].strip():
                documents.append(
                    Document(
                        page_content=splits[0].strip(),
                        metadata={
                            "source": file_name,
                            "file_path": os.path.abspath(pdf_path),
                            "page": 1,
                        },
                    )
                )

            for i in range(1, len(splits), 2):
                page_num = int(splits[i])
                page_text = splits[i + 1].strip() if i + 1 < len(splits) else ""
                if page_text:
                    documents.append(
                        Document(
                            page_content=page_text,
                            metadata={
                                "source": file_name,
                                "file_path": os.path.abspath(pdf_path),
                                "page": page_num,
                            },
                        )
                    )
        else:
            # 페이지 구분자가 없는 경우 전체를 1페이지로 처리
            if extracted_text.strip():
                documents.append(
                    Document(
                        page_content=extracted_text.strip(),
                        metadata={
                            "source": file_name,
                            "file_path": os.path.abspath(pdf_path),
                            "page": 1,
                        },
                    )
                )

        # 만약 OCR 결과가 비어있을 경우 fallback으로 pypdf 시도
        if not documents:
            print("  [경고] Gemini OCR 결과 텍스트가 비어 있어 기본 PDF 텍스트 추출기를 시도합니다.")
            try:
                reader = pypdf.PdfReader(pdf_path)
                for page_idx, page in enumerate(reader.pages):
                    t = page.extract_text() or ""
                    if t.strip():
                        documents.append(
                            Document(
                                page_content=t.strip(),
                                metadata={
                                    "source": file_name,
                                    "file_path": os.path.abspath(pdf_path),
                                    "page": page_idx + 1,
                                },
                            )
                        )
            except Exception:
                pass

        total_pages = max(len(documents), pypdf_pages)
        for d in documents:
            d.metadata["total_pages"] = total_pages

        total_chars = sum(len(d.page_content) for d in documents)
        pages_content = [
            {
                "page": d.metadata.get("page", idx + 1),
                "text": d.page_content,
                "char_count": len(d.page_content),
            }
            for idx, d in enumerate(documents)
        ]

        sample_preview = ""
        print(f"  [출력값] 파일명: {file_name} ({file_size_kb:.1f} KB)")
        print(f"  [출력값] OCR 추출 성공: {len(documents)}개 페이지 구획, 총 글자 수: {total_chars:,}자")
        if documents:
            sample_preview = documents[0].page_content.strip().replace("\n", " ")[:150]
            print(f"  [출력값] 1페이지 OCR 내용 미리보기(앞 150자): \"{sample_preview}...\"")
        print(f"  [소요 시간] {elapsed:.2f}초")

        return {
            "file_name": file_name,
            "file_size_kb": round(file_size_kb, 2),
            "total_pages": len(documents) if documents else total_pages,
            "total_chars": total_chars,
            "sample_preview": sample_preview,
            "pages_content": pages_content,
            "ocr_prompt": prompt_to_use,
            "elapsed": round(elapsed, 2),
            "documents": documents,
        }

    def _step_a2_split_chunks(self, documents: List[Document]) -> Dict[str, Any]:
        """스텝 A2. 청크로 나누기 (작은 원문 조각)"""
        self._print_step("스텝 A2", "청크로 나누기 (작은 원문 조각)")
        print(
            f"  [설정값] chunk_size={self.config.chunk_size}, chunk_overlap={self.config.chunk_overlap}"
        )
        start_time = time.time()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        chunks = splitter.split_documents(documents)
        elapsed = time.time() - start_time

        # 통계 계산
        lengths = [len(c.page_content) for c in chunks] if chunks else [0]
        avg_len = sum(lengths) / len(lengths) if lengths else 0

        # 출력값 확인
        sample_first = ""
        sample_last = ""
        print(f"  [출력값] 분할된 총 청크 수: {len(chunks)}개")
        print(f"  [출력값] 청크 길이 통계: 최소 {min(lengths)}자 / 최대 {max(lengths)}자 / 평균 {avg_len:.1f}자")
        if chunks:
            sample_first = chunks[0].page_content.strip().replace("\n", " ")[:120]
            print(f"  [출력값] [청크 #1] (페이지 {chunks[0].metadata.get('page')}, {len(chunks[0].page_content)}자):")
            print(f"          \"{sample_first}...\"")
            if len(chunks) > 1:
                sample_last = chunks[-1].page_content.strip().replace("\n", " ")[:120]
                print(f"  [출력값] [청크 #{len(chunks)}] (페이지 {chunks[-1].metadata.get('page')}, {len(chunks[-1].page_content)}자):")
                print(f"          \"{sample_last}...\"")
        print(f"  [소요 시간] {elapsed:.2f}초")

        # 청크별 상세 메타데이터 목록 생성
        chunk_details = []
        for idx, c in enumerate(chunks):
            chunk_details.append(
                {
                    "chunk_index": idx,
                    "page": c.metadata.get("page", 1),
                    "char_count": len(c.page_content),
                    "content": c.page_content,
                    "preview": c.page_content.strip().replace("\n", " ")[:120],
                }
            )

        return {
            "chunk_count": len(chunks),
            "min_len": min(lengths),
            "max_len": max(lengths),
            "avg_len": round(avg_len, 1),
            "sample_first": sample_first,
            "sample_last": sample_last,
            "elapsed": round(elapsed, 2),
            "chunks": chunks,
            "chunk_details": chunk_details,
        }

    def _step_a3_embed_chunks(self, chunks: List[Document]) -> Dict[str, Any]:
        """스텝 A3. 임베딩하기 (청크 -> 숫자 배열, SQLite 임베딩 캐시 적용)"""
        self._print_step("스텝 A3", "임베딩하기 (청크 -> 숫자 배열, 영구 캐시 가속)")
        print(f"  [설정값] 임베딩 모델: {self.config.embedding_model}")
        start_time = time.time()

        texts = [c.page_content for c in chunks]

        # 1. 로컬 SQLite 임베딩 영구 캐시 조회
        cached_hits, missing_items = self.vector_store.get_cached_embeddings(
            model_name=self.config.embedding_model,
            texts=texts,
        )
        hit_count = len(cached_hits)
        miss_count = len(missing_items)

        print(f"  [캐시 확인] 총 {len(texts)}개 청크 중 캐시 적중(Hit): {hit_count}개, 신규 생성 필요(Miss): {miss_count}개")

        # 2. 캐시 미스 항목만 Gemini API 호출
        if missing_items:
            missing_texts = [item[1] for item in missing_items]
            print(f"  [진행] {miss_count}개 신규 청크에 대해 Gemini 임베딩 API 요청 중...")
            try:
                new_embeddings = self.embeddings.embed_documents(missing_texts)
            except Exception as e:
                diag = diagnose_gemini_error(e)
                if diag.get("is_429"):
                    print_gemini_429_diagnosis(e)
                raise

            # 3. 신규 생성된 임베딩을 캐시에 저장
            self.vector_store.save_cached_embeddings(
                model_name=self.config.embedding_model,
                items=list(zip(missing_texts, new_embeddings)),
            )

            for (orig_idx, _), emb in zip(missing_items, new_embeddings):
                cached_hits[orig_idx] = emb
        else:
            print("  ⚡ [캐시 100% 적중] 모든 청크의 임베딩이 캐시되어 있어 Gemini API 호출을 생략했습니다! (API 비용 0원)")

        # 원본 인덱스 순서대로 embeddings 복원
        embeddings = [cached_hits[i] for i in range(len(texts))]
        elapsed = time.time() - start_time

        vector_dim = len(embeddings[0]) if embeddings else 0
        sample_vector = [round(v, 5) for v in embeddings[0][:5]] if embeddings else []

        print(f"  [출력값] 임베딩 완료된 벡터 수: {len(embeddings)}개 (캐시 적중: {hit_count}개, 신규 생성: {miss_count}개)")
        print(f"  [출력값] 임베딩 벡터 차원(Dimension): {vector_dim}차원")
        if embeddings:
            print(f"  [출력값] 청크 #1 벡터 샘플 (앞 5개 차원): {sample_vector} ...")
        print(f"  [소요 시간] {elapsed:.2f}초")

        return {
            "vector_count": len(embeddings),
            "dimension": vector_dim,
            "sample_vector": sample_vector,
            "elapsed": round(elapsed, 2),
            "embeddings": embeddings,
            "cache_hit_count": hit_count,
            "cache_miss_count": miss_count,
        }

    def _step_a4_save_to_sqlite(
        self, chunks: List[Document], embeddings: List[List[float]]
    ) -> Dict[str, Any]:
        """스텝 A4. 벡터 스토어 저장 (SQLite DB: 벡터 + 원문 + 출처 연결)"""
        self._print_step("스텝 A4", "벡터 스토어 저장 (SQLite DB)")
        print(f"  [설정값] SQLite DB 경로: {self.config.db_path}")
        start_time = time.time()

        saved_count = self.vector_store.add_documents(chunks, embeddings)
        total_in_db = self.vector_store.count()
        file_size_kb = (
            os.path.getsize(self.config.db_path) / 1024
            if os.path.exists(self.config.db_path)
            else 0
        )
        elapsed = time.time() - start_time

        # 출력값 확인
        print(f"  [출력값] 이번 작업으로 저장된 레코드 수: {saved_count}개")
        print(f"  [출력값] SQLite DB 내 총 청크 수: {total_in_db}개 (파일 크기: {file_size_kb:.2f} KB)")
        print(f"  [출력값] 저장 완료: [벡터(BLOB) + 원문 텍스트 + 출처/페이지 메타데이터] 연결 완료")
        print(f"  [소요 시간] {elapsed:.2f}초")

        return {
            "saved_count": saved_count,
            "total_in_db": total_in_db,
            "db_path": self.config.db_path,
            "file_size_kb": round(file_size_kb, 2),
            "elapsed": round(elapsed, 2),
        }

    # =========================================================================
    # B. 질문에 답하기 (런타임: 질문이 들어올 때마다)
    # =========================================================================

    def ask(
        self,
        question: str,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
        llm_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """B 단계 전체 실행: 질문 입력 -> 관련 청크 검색 -> 프롬프트 구성 -> LLM 답변"""
        self._print_header("B. 질문에 답하기 (질문이 들어올 때마다 · 런타임)")

        # ---------------------------------------------------------
        # [스텝 B1] 질문 입력 (질문도 임베딩)
        # ---------------------------------------------------------
        step1_output = self._step_b1_embed_query(question)

        # ---------------------------------------------------------
        # [스텝 B2] 관련 청크 검색 (숫자 비교 -> 원문 반환: 하이브리드 RRF)
        # ---------------------------------------------------------
        step2_output = self._step_b2_search_chunks(
            step1_output["query_vector"], query_text=question, top_k=top_k
        )

        # ---------------------------------------------------------
        # [스텝 B3] 프롬프트 구성 (질문 + 원문 + 지시문)
        # ---------------------------------------------------------
        step3_output = self._step_b3_build_prompt(
            question=question, search_results=step2_output["retrieved_docs"]
        )

        # ---------------------------------------------------------
        # [스텝 B4] LLM -> 답변 (근거를 읽고 생성)
        # ---------------------------------------------------------
        step4_output = self._step_b4_generate_answer(
            messages=step3_output["messages"],
            llm_model=llm_model,
            temperature=temperature,
        )

        return {
            "question": question,
            "step1_query": step1_output,
            "step2_retrieval": step2_output,
            "step3_prompt": step3_output,
            "step4_generation": step4_output,
            "answer": step4_output["answer"],
        }

    def _step_b1_embed_query(self, question: str) -> Dict[str, Any]:
        """스텝 B1. 질문 입력 (질문도 임베딩, 캐시 적용)"""
        self._print_step("스텝 B1", "질문 입력 (질문도 임베딩)")
        print(f"  [출력값] 사용자 질문: \"{question}\"")
        start_time = time.time()

        cached_hits, _ = self.vector_store.get_cached_embeddings(
            model_name=self.config.embedding_model,
            texts=[question],
        )
        if 0 in cached_hits:
            query_vector = cached_hits[0]
            print("  ⚡ [질의어 캐시 적중] 캐시된 질문 임베딩 벡터를 재사용합니다 (API 호출 0원).")
        else:
            try:
                query_vector = self.embeddings.embed_query(question)
                self.vector_store.save_cached_embeddings(
                    model_name=self.config.embedding_model,
                    items=[(question, query_vector)],
                )
            except Exception as e:
                diag = diagnose_gemini_error(e)
                if diag.get("is_429"):
                    print_gemini_429_diagnosis(e)
                raise

        elapsed = time.time() - start_time
        sample_vector = [round(v, 5) for v in query_vector[:5]]
        print(f"  [출력값] 질문 임베딩 완료: {len(query_vector)}차원 벡터 생성")
        print(f"  [출력값] 질문 벡터 샘플 (앞 5개 차원): {sample_vector} ...")
        print(f"  [소요 시간] {elapsed:.2f}초")

        return {
            "question": question,
            "dimension": len(query_vector),
            "sample_vector": sample_vector,
            "elapsed": round(elapsed, 2),
            "query_vector": query_vector,
        }

    def _step_b2_search_chunks(
        self,
        query_vector: List[float],
        query_text: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """스텝 B2. 관련 청크 검색 (하이브리드: Dense 코사인 유사도 + Sparse BM25 키워드 RRF 융합)"""
        k = top_k if top_k is not None else self.config.top_k
        self._print_step("스텝 B2", "관련 청크 검색 (하이브리드: 벡터 + 키워드 RRF)")
        print(f"  [설정값] 검색 상위 청크 수(top_k): {k}")
        if query_text:
            print(f"  [검색방식] 하이브리드 RRF (Dense 벡터 + Sparse 키워드 매칭)")
        start_time = time.time()

        search_results = self.vector_store.similarity_search_by_vector(
            query_vector, top_k=k, query_text=query_text
        )
        elapsed = time.time() - start_time

        results_preview = []
        print(f"  [출력값] 검색된 청크 수: {len(search_results)}개")
        for rank, (doc, score) in enumerate(search_results, start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            chunk_idx = doc.metadata.get("chunk_index", "?")
            dense_rank = doc.metadata.get("dense_rank")
            bm25_rank = doc.metadata.get("bm25_rank")
            dense_sim = doc.metadata.get("dense_similarity")
            bm25_sc = doc.metadata.get("bm25_score")
            preview = doc.page_content.strip().replace("\n", " ")[:100]
            results_preview.append(
                {
                    "rank": rank,
                    "score": round(score, 4),
                    "source": source,
                    "page": page,
                    "chunk_index": chunk_idx,
                    "dense_rank": dense_rank,
                    "bm25_rank": bm25_rank,
                    "dense_similarity": dense_sim,
                    "bm25_score": bm25_sc,
                    "preview": preview,
                    "full_content": doc.page_content,
                }
            )
            extra_info = ""
            if dense_rank is not None and bm25_rank is not None:
                extra_info = f" [벡터: #{dense_rank}({dense_sim}), 키워드: #{bm25_rank}({bm25_sc})]"
            print(f"  [출력값] #{rank} [융합점수: {score:.4f}] 출처: {source} (p.{page}, 청크 #{chunk_idx}){extra_info}")
            print(f"          내용: \"{preview}...\"")
        print(f"  [소요 시간] {elapsed:.2f}초")

        return {
            "retrieved_count": len(search_results),
            "results_preview": results_preview,
            "elapsed": round(elapsed, 2),
            "retrieved_docs": search_results,
        }

    def _step_b3_build_prompt(
        self, question: str, search_results: List[Tuple[Document, float]]
    ) -> Dict[str, Any]:
        """스텝 B3. 프롬프트 구성 (질문 + 원문 + 지시문: 원래 질문도 그대로 전달)"""
        self._print_step("스텝 B3", "프롬프트 구성 (질문 + 원문 + 지시문)")
        start_time = time.time()

        # 검색된 원문 컨텍스트 조합
        context_parts = []
        for idx, (doc, score) in enumerate(search_results, start=1):
            page = doc.metadata.get("page", "?")
            source = doc.metadata.get("source", "PDF")
            context_parts.append(
                f"[참고자료 #{idx} (출처: {source}, 페이지: {page})]\n{doc.page_content}"
            )
        context_str = "\n\n".join(context_parts)

        # 시스템 지시문과 템플릿
        system_instruction = (
            "당신은 주어진 참고 문서를 바탕으로 사용자의 질문에 정확하게 답변하는 전문 AI입니다.\n"
            "반드시 아래 [참고자료]의 내용에 근거하여 사실에 입각해 답변하세요.\n"
            "참고자료에 없는 내용은 추측하지 말고 솔직하게 모른다고 답변하세요.\n"
            "답변 끝에는 참조한 페이지 번호(출처)를 명시해 주세요."
        )

        user_content = (
            f"=== [참고자료] ===\n{context_str}\n\n"
            f"=== [사용자 질문] ===\n{question}\n\n"
            f"위 참고자료를 바탕으로 사용자의 질문에 명확하고 친절하게 답변해 주세요."
        )

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_instruction),
                (
                    "user",
                    "=== [참고자료] ===\n{context}\n\n"
                    "=== [사용자 질문] ===\n{question}\n\n"
                    "위 참고자료를 바탕으로 사용자의 질문에 명확하고 친절하게 답변해 주세요.",
                ),
            ]
        )

        prompt_value = prompt_template.invoke(
            {"context": context_str, "question": question}
        )
        messages = prompt_value.to_messages()
        elapsed = time.time() - start_time

        print(f"  [출력값] 프롬프트 구성 요소:")
        print(f"          - 시스템 지시문: {len(system_instruction)}자")
        print(f"          - 결합된 참고자료: {len(search_results)}개 청크 ({len(context_str)}자)")
        print(f"          - 원래 질문 그대로 전달: \"{question}\"")
        print(f"  [출력값] --- 최종 전달 프롬프트 미리보기 ---")
        preview_prompt = user_content[:250].replace("\n", " ")
        print(f"          {preview_prompt}...")

        return {
            "messages": messages,
            "context_str": context_str,
            "system_instruction": system_instruction,
            "user_content": user_content,
            "full_prompt_text": prompt_value.to_string(),
            "elapsed": round(elapsed, 2),
        }

    def _step_b4_generate_answer(
        self,
        messages: List[Any],
        llm_model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """스텝 B4. LLM -> 답변 (근거를 읽고 생성)"""
        chosen_model = llm_model or self.config.llm_model
        temp = temperature if temperature is not None else self.config.temperature

        self._print_step("스텝 B4", "LLM -> 답변 (근거를 읽고 생성)")
        print(f"  [설정값] LLM 모델: {chosen_model} (temperature: {temp})")
        start_time = time.time()

        # 설정된 기본 모델과 동일하면 self.llm 재사용, 다르면 동적 인스턴스 생성
        if chosen_model == self.config.llm_model and (temperature is None or temperature == self.config.temperature):
            target_llm = self.llm
        else:
            target_llm = self._create_llm(chosen_model, temp)

        try:
            response = target_llm.invoke(messages)
        except Exception as e:
            diag = diagnose_gemini_error(e)
            if diag.get("is_429"):
                print_gemini_429_diagnosis(e)
            raise
        elapsed = time.time() - start_time

        # response.content가 str 또는 블록 리스트([{'type': 'text', ...}])일 수 있으므로 정제
        raw_content = response.content
        if isinstance(raw_content, list):
            text_parts = []
            for item in raw_content:
                if isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
                elif isinstance(item, str):
                    text_parts.append(item)
                elif hasattr(item, "text"):
                    text_parts.append(getattr(item, "text"))
            answer_text = "".join(text_parts) if text_parts else str(raw_content)
        else:
            answer_text = str(raw_content)

        # 모델 응답에 <think>...</think> 또는 <thought>...</thought> 블록이 있을 경우 후처리
        answer_text = re.sub(r"<(think|thought)>[\s\S]*?</\1>", "", answer_text).strip()

        print(f"  [출력값] LLM({chosen_model}) 답변 생성 완료 ({len(answer_text)}자, 소요시간: {elapsed:.2f}초)")
        print(f"\n{'=' * 60}")
        print(" [최종 생성된 답변]")
        print(f"{'=' * 60}")
        print(answer_text)
        print(f"{'=' * 60}\n")

        return {
            "answer": answer_text,
            "latency": round(elapsed, 2),
            "model": chosen_model,
        }

    # =========================================================================
    # 콘솔 포맷팅 헬퍼 함수
    # =========================================================================

    @staticmethod
    def _print_header(title: str):
        print("\n" + "#" * 70)
        print(f" ## {title}")
        print("#" * 70)

    @staticmethod
    def _print_step(step_id: str, title: str):
        print("\n" + "-" * 60)
        print(f" ▶ [{step_id}] {title}")
        print("-" * 60)
