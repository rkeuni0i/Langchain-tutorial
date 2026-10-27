# PDF 기반 LangChain RAG 시스템

Gemini 임베딩(`gemini-embedding-2`), 멀티모달 비전 OCR 및 Gemini LLM(`gemini-3.8-flash`), 그리고 별도 설치가 필요 없는 SQLite 기반 경량 벡터 스토어를 활용한 RAG(Retrieval-Augmented Generation) 파이프라인 및 시각화 대시보드입니다.

---

## 📁 폴더 구성

```text
rag/
├── rag_app.py              # FastAPI 기반 웹 대시보드 애플리케이션 (포트 8000)
├── rag_pipeline.py         # A1~A4 (인덱싱), B1~B4 (런타임 질의응답) 전체 RAG 파이프라인
├── sqlite_vector_store.py  # SQLite + NumPy 기반 경량 벡터 스토어 (BLOB + 코사인 유사도)
├── run_rag.py              # CLI 환경용 단계별 실행 스크립트
├── make_sample_pdf.py      # 테스트용 샘플 PDF 생성기
├── sample_knowledge.pdf    # 테스트용 기본 샘플 PDF 문서
├── rag_sqlite.db           # 인덱싱된 청크 및 임베딩 벡터 저장 DB
├── uploads/                # 업로드된 PDF 파일 및 캐시 파일 디렉토리
│   ├── last_indexed_info.json
│   └── ...
└── __init__.py             # Python 패키지 초기화 파일
```

---

## 🚀 실행 방법

프로젝트 루트 디렉토리 또는 `rag/` 폴더 어디에서나 실행 가능합니다.

### 1. 웹 대시보드 실행 (추천)
```bash
# 프로젝트 루트에서 실행 시
python rag/rag_app.py

# 또는 rag 폴더로 이동 후 실행 시
cd rag
python rag_app.py
```
- 브라우저 접속: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Swagger API 명세: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 2. 콘솔 CLI 모드로 실행
```bash
# 단일 질문 실행
python rag/run_rag.py --query "RAG에서 텍스트 분할시 분할의 단위를 뭐라고 부르는가?"

# 대화형 모드(Interactive Mode) 실행
python rag/run_rag.py --interactive
```

