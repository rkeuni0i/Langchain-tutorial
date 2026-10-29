"""RAG 파이프라인 실행 스크립트

사용자가 chunk_size, chunk_overlap, top_k 등의 설정값을 직접 지정하고
이미지 다이어그램에 따른 각 스텝별 출력값을 상세히 확인할 수 있습니다.
"""

import argparse
import os
import sys
from pathlib import Path

# 현재 디렉토리 sys.path 보정
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Windows 콘솔 utf-8 출력 인코딩 보장
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from .rag_pipeline import PDFRAGPipeline, RAGConfig
    from .make_sample_pdf import create_sample_pdf
    from .gemini_error_debugger import diagnose_gemini_error, print_gemini_429_diagnosis
except ImportError:
    from rag_pipeline import PDFRAGPipeline, RAGConfig
    from make_sample_pdf import create_sample_pdf
    from gemini_error_debugger import diagnose_gemini_error, print_gemini_429_diagnosis


def parse_args():
    parser = argparse.ArgumentParser(
        description="LangChain & SQLite 기반 PDF RAG 단계별 실행기"
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=str(CURRENT_DIR / "sample_knowledge.pdf"),
        help="입력할 PDF 파일 경로 (기본값: rag/sample_knowledge.pdf)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="텍스트 분할 청크 크기 (기본값: 500)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="청크 간 중복 글자 수 (기본값: 100)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="검색할 관련 청크 수 (기본값: 2)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="RAG의 주요 이점과 청크 오버랩(Chunk overlap)의 역할은 무엇인가요?",
        help="질문 내용",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(CURRENT_DIR / "rag_sqlite.db"),
        help="SQLite 벡터 데이터베이스 파일 경로 (기본값: rag/rag_sqlite.db)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="gemini-embedding-2",
        help="Gemini 임베딩 모델명 (기본값: gemini-embedding-2)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gemini-3.8-flash",
        choices=["gemini-3.8-flash", "gemma4:e2b"],
        help="답변 생성 LLM 모델 (gemini-3.8-flash 또는 gemma4:e2b, 기본값: gemini-3.8-flash)",
    )
    parser.add_argument(
        "--ocr-prompt",
        type=str,
        default=None,
        help="Gemini OCR 추출 시 사용할 사용자 정의 프롬프트 (기본값: 내장 표준 프롬프트)",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="이미 인덱싱된 DB를 사용하고 인덱싱 단계 건너뛰기",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="인덱싱 완료 후 질문을 계속 입력받는 대화형 모드 실행",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("      LangChain PDF RAG 파이프라인 (스텝별 출력 확인)")
    print("=" * 70)
    print(f" • PDF 문서 경로: {args.pdf}")
    print(f" • 사용자 설정 chunk_size   : {args.chunk_size}")
    print(f" • 사용자 설정 chunk_overlap: {args.chunk_overlap}")
    print(f" • 검색 청크 개수(top_k)    : {args.top_k}")
    print(f" • SQLite DB 경로          : {args.db_path}")
    print(f" • 임베딩 모델             : {args.embedding_model}")
    print(f" • 답변 생성 LLM           : {args.llm_model}")
    print("=" * 70)

    # 지정된 PDF 파일이 없을 경우 샘플 PDF 자동 생성
    if not os.path.exists(args.pdf):
        print(f"\n[알림] 지정된 PDF 파일('{args.pdf}')이 없어 테스트용 샘플 PDF를 자동 생성합니다.")
        create_sample_pdf(args.pdf)

    # 1. RAG 파이프라인 설정 및 인스턴스화
    config = RAGConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        db_path=args.db_path,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        temperature=0.2,
    )
    pipeline = PDFRAGPipeline(config=config)

    try:
        # 2. [A 단계: 미리 준비하기] 인덱싱 실행 (스텝 A1 ~ A4)
        if not args.skip_index:
            pipeline.index_pdf(
                pdf_path=args.pdf,
                clear_existing=True,
                ocr_prompt=args.ocr_prompt,
            )
        else:
            print("\n[알림] --skip-index 옵션에 따라 기존 SQLite DB를 그대로 사용합니다.")

        # 3. [B 단계: 질문에 답하기] 런타임 질의응답 실행 (스텝 B1 ~ B4)
        if args.interactive:
            print("\n" + "=" * 70)
            print(" [대화형 모드 시작] PDF 내용에 대해 무엇이든 질문하세요! (종료: 'q' 또는 'exit')")
            print("=" * 70)
            while True:
                try:
                    user_q = input("\n[질문 입력] > ").strip()
                    if not user_q:
                        continue
                    if user_q.lower() in ("q", "exit", "quit"):
                        print("대화형 모드를 종료합니다.")
                        break
                    pipeline.ask(question=user_q)
                except KeyboardInterrupt:
                    print("\n종료합니다.")
                    break
                except Exception as loop_e:
                    diag = diagnose_gemini_error(loop_e)
                    if diag.get("is_429"):
                        print_gemini_429_diagnosis(loop_e)
                    else:
                        print(f"[오류 발생] {loop_e}")
        else:
            # 단일 질문 실행
            pipeline.ask(question=args.query)
    except Exception as e:
        diag = diagnose_gemini_error(e)
        if diag.get("is_429"):
            print_gemini_429_diagnosis(e)
            sys.exit(1)
        else:
            raise


if __name__ == "__main__":
    main()

