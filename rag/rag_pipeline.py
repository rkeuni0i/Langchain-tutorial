"""PDF 기반 LangChain RAG 파이프라인

다이어그램 구조(A. 미리 준비하기 / B. 질문에 답하기)의 
각 스텝별 출력값을 콘솔에서 투명하게 확인할 수 있도록 구현되었습니다.
"""

import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from .sqlite_vector_store import SQLiteVectorStore
except ImportError:
    from sqlite_vector_store import SQLiteVectorStore


@dataclass
class RAGConfig:
    """사용자가 변경 가능한 RAG 설정값 클래스"""

    # A. 청크 분할 설정
    chunk_size: int = 800  # 청크 분할 크기 (사용자 변경 가능)
    chunk_overlap: int = 150  # 청크 간 중복 글자 수 (사용자 변경 가능)

    # B. 벡터 스토어 및 검색 설정
    db_path: str = str(CURRENT_DIR / "rag_sqlite.db")  # SQLite 파일 경로 (절대경로)
    top_k: int = 3  # 검색할 관련 청크 수

    # C. 모델 설정
    embedding_model: str = "gemini-embedding-2"  # Gemini 2세대 임베딩 모델
    llm_model: str = "gemini-3.8-flash"  # Gemini LLM 모델
    temperature: float = 0.2  # LLM 응답 창의성 (0.0: 보수적/정확, 1.0: 창의적)


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

        # 3. LLM 모델 초기화
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.llm_model,
            google_api_key=self.api_key,
            temperature=self.config.temperature,
            timeout=30,
            max_retries=3,
            streaming=False,
        )

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
        """인덱싱된 문서의 페이지별 전체 내용 조회 (캐시 우선, 없으면 SQLite DB에서 복원)"""
        if self.last_indexed_info and self.last_indexed_info.get("pages_content"):
            return {
                "available": True,
                "file_name": self.last_indexed_info.get("file_name", "PDF 문서"),
                "total_pages": self.last_indexed_info.get(
                    "total_pages", len(self.last_indexed_info["pages_content"])
                ),
                "total_chars": self.last_indexed_info.get("total_chars", 0),
                "pages": self.last_indexed_info["pages_content"],
            }

        # DB에 데이터가 있으나 캐시가 없는 경우 SQLite DB에서 페이지별 텍스트 복원
        try:
            if self.vector_store.count() == 0:
                return {"available": False, "pages": []}

            with self.vector_store._get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT DISTINCT source_file FROM rag_chunks LIMIT 1;")
                row = cur.fetchone()
                file_name = row["source_file"] if row else "PDF 문서"

                cur.execute(
                    "SELECT page_number, content FROM rag_chunks ORDER BY page_number, chunk_index;"
                )
                rows = cur.fetchall()

                pages_dict: Dict[int, List[str]] = {}
                for r in rows:
                    p = r["page_number"]
                    if p not in pages_dict:
                        pages_dict[p] = []
                    pages_dict[p].append(r["content"])

                pages = []
                total_chars = 0
                for p in sorted(pages_dict.keys()):
                    p_text = "\n\n".join(pages_dict[p])
                    total_chars += len(p_text)
                    pages.append(
                        {
                            "page": p,
                            "text": p_text,
                            "char_count": len(p_text),
                        }
                    )

                self.last_indexed_info = {
                    "file_name": file_name,
                    "total_pages": len(pages),
                    "total_chars": total_chars,
                    "pages_content": pages,
                }
                self._save_cache()

                return {
                    "available": True,
                    "file_name": file_name,
                    "total_pages": len(pages),
                    "total_chars": total_chars,
                    "pages": pages,
                }
        except Exception as e:
            return {"available": False, "pages": [], "error": str(e)}

    # =========================================================================
    # A. 미리 준비하기 (인덱싱: 자료를 등록하거나 변경할 때)
    # =========================================================================

    def index_pdf(self, pdf_path: str, clear_existing: bool = True) -> Dict[str, Any]:
        """A 단계 전체 실행: PDF 읽기 -> 청크 분할 -> 임베딩 -> SQLite 저장"""
        self._print_header("A. 미리 준비하기 (자료를 등록하거나 변경할 때 · 인덱싱)")

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

        if clear_existing:
            self.vector_store.clear()
            self.clear_cache()
            print("  ℹ 기존 SQLite 벡터 스토어 데이터를 초기화했습니다.")

        # ---------------------------------------------------------
        # [스텝 A1] 문서 읽기 (PDF -> 텍스트)
        # ---------------------------------------------------------
        step1_output = self._step_a1_load_pdf(pdf_path)

        # ---------------------------------------------------------
        # [스텝 A2] 청크로 나누기 (작은 원문 조각)
        # ---------------------------------------------------------
        step2_output = self._step_a2_split_chunks(step1_output["documents"])

        # ---------------------------------------------------------
        # [스텝 A3] 임베딩하기 (청크 -> 숫자 배열)
        # ---------------------------------------------------------
        step3_output = self._step_a3_embed_chunks(step2_output["chunks"])

        # ---------------------------------------------------------
        # [스텝 A4] 벡터 스토어 저장 (SQLite DB: 벡터 + 원문 + 출처 연결)
        # ---------------------------------------------------------
        step4_output = self._step_a4_save_to_sqlite(
            step2_output["chunks"], step3_output["embeddings"]
        )

        # 최근 인덱싱된 문서 정보 캐시 업데이트 및 저장
        self.last_indexed_info = {
            "file_name": step1_output["file_name"],
            "total_pages": step1_output["total_pages"],
            "total_chars": step1_output["total_chars"],
            "pages_content": step1_output["pages_content"],
            "elapsed": step1_output["elapsed"],
        }
        self._save_cache()

        return {
            "step1_load": step1_output,
            "step2_split": step2_output,
            "step3_embed": step3_output,
            "step4_store": step4_output,
        }

    def _step_a1_load_pdf(self, pdf_path: str) -> Dict[str, Any]:
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
        ocr_prompt = (
            "다음 제공된 PDF 문서에 포함된 모든 글자(텍스트)를 누락 없이 있는 그대로 OCR 추출해 주세요.\n\n"
            "[엄격한 준수 사항]\n"
            "1. 사진, 일러스트, 그래프, 다이어그램 등 이미지에 대한 설명이나 시각적 묘사는 모두 빼고 완전히 제외하세요.\n"
            "2. 오직 문서에 적혀 있는 순수한 문자/텍스트(활자) 내용만을 그대로 추출하세요.\n"
            "3. 각 페이지의 시작 부분에 반드시 '--- [PAGE N] ---' (예: --- [PAGE 1] ---) 형식의 페이지 구분자를 넣어주세요.\n"
            "4. 부가적인 인사말, 머리말, 마크다운 코드 블록(```) 등의 래핑 없이 오직 추출된 텍스트만 출력하세요."
        )

        print(f"  [진행] Gemini 3.8 Flash 멀티모달 비전을 통한 PDF 텍스트 OCR 추출 중...")
        response = self.genai_client.models.generate_content(
            model="gemini-3.8-flash",
            contents=[
                types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                ocr_prompt,
            ],
        )
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

        return {
            "chunk_count": len(chunks),
            "min_len": min(lengths),
            "max_len": max(lengths),
            "avg_len": round(avg_len, 1),
            "sample_first": sample_first,
            "sample_last": sample_last,
            "elapsed": round(elapsed, 2),
            "chunks": chunks,
        }

    def _step_a3_embed_chunks(self, chunks: List[Document]) -> Dict[str, Any]:
        """스텝 A3. 임베딩하기 (청크 -> 숫자 배열)"""
        self._print_step("스텝 A3", "임베딩하기 (청크 -> 숫자 배열)")
        print(f"  [설정값] 임베딩 모델: {self.config.embedding_model}")
        start_time = time.time()

        texts = [c.page_content for c in chunks]
        print(f"  [진행] {len(texts)}개 청크에 대해 Gemini 임베딩 API 요청 중...")
        embeddings = self.embeddings.embed_documents(texts)
        elapsed = time.time() - start_time

        vector_dim = len(embeddings[0]) if embeddings else 0
        sample_vector = [round(v, 5) for v in embeddings[0][:5]] if embeddings else []

        # 출력값 확인
        print(f"  [출력값] 임베딩 완료된 벡터 수: {len(embeddings)}개")
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

    def ask(self, question: str) -> Dict[str, Any]:
        """B 단계 전체 실행: 질문 입력 -> 관련 청크 검색 -> 프롬프트 구성 -> LLM 답변"""
        self._print_header("B. 질문에 답하기 (질문이 들어올 때마다 · 런타임)")

        # ---------------------------------------------------------
        # [스텝 B1] 질문 입력 (질문도 임베딩)
        # ---------------------------------------------------------
        step1_output = self._step_b1_embed_query(question)

        # ---------------------------------------------------------
        # [스텝 B2] 관련 청크 검색 (숫자 비교 -> 원문 반환)
        # ---------------------------------------------------------
        step2_output = self._step_b2_search_chunks(step1_output["query_vector"])

        # ---------------------------------------------------------
        # [스텝 B3] 프롬프트 구성 (질문 + 원문 + 지시문)
        # ---------------------------------------------------------
        step3_output = self._step_b3_build_prompt(
            question=question, search_results=step2_output["retrieved_docs"]
        )

        # ---------------------------------------------------------
        # [스텝 B4] LLM -> 답변 (근거를 읽고 생성)
        # ---------------------------------------------------------
        step4_output = self._step_b4_generate_answer(step3_output["messages"])

        return {
            "question": question,
            "step1_query": step1_output,
            "step2_retrieval": step2_output,
            "step3_prompt": step3_output,
            "step4_generation": step4_output,
            "answer": step4_output["answer"],
        }

    def _step_b1_embed_query(self, question: str) -> Dict[str, Any]:
        """스텝 B1. 질문 입력 (질문도 임베딩)"""
        self._print_step("스텝 B1", "질문 입력 (질문도 임베딩)")
        print(f"  [출력값] 사용자 질문: \"{question}\"")
        start_time = time.time()

        query_vector = self.embeddings.embed_query(question)
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

    def _step_b2_search_chunks(self, query_vector: List[float]) -> Dict[str, Any]:
        """스텝 B2. 관련 청크 검색 (숫자 비교 -> 원문 반환: 코사인 유사도)"""
        self._print_step("스텝 B2", "관련 청크 검색 (숫자 비교 -> 원문 반환)")
        print(f"  [설정값] 검색 상위 청크 수(top_k): {self.config.top_k}")
        start_time = time.time()

        search_results = self.vector_store.similarity_search_by_vector(
            query_vector, top_k=self.config.top_k
        )
        elapsed = time.time() - start_time

        results_preview = []
        print(f"  [출력값] 검색된 청크 수: {len(search_results)}개")
        for rank, (doc, score) in enumerate(search_results, start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            chunk_idx = doc.metadata.get("chunk_index", "?")
            preview = doc.page_content.strip().replace("\n", " ")[:100]
            results_preview.append(
                {
                    "rank": rank,
                    "score": round(score, 4),
                    "source": source,
                    "page": page,
                    "chunk_index": chunk_idx,
                    "preview": preview,
                    "full_content": doc.page_content,
                }
            )
            print(f"  [출력값] #{rank} [유사도: {score:.4f}] 출처: {source} (p.{page}, 청크 #{chunk_idx})")
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

    def _step_b4_generate_answer(self, messages: List[Any]) -> Dict[str, Any]:
        """스텝 B4. LLM -> 답변 (근거를 읽고 생성)"""
        self._print_step("스텝 B4", "LLM -> 답변 (근거를 읽고 생성)")
        print(f"  [설정값] LLM 모델: {self.config.llm_model} (temperature: {self.config.temperature})")
        start_time = time.time()

        response = self.llm.invoke(messages)
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

        print(f"  [출력값] LLM 답변 생성 완료 ({len(answer_text)}자, 소요시간: {elapsed:.2f}초)")
        print(f"\n{'=' * 60}")
        print(" [최종 생성된 답변]")
        print(f"{'=' * 60}")
        print(answer_text)
        print(f"{'=' * 60}\n")

        return {
            "answer": answer_text,
            "latency": round(elapsed, 2),
            "model": self.config.llm_model,
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
