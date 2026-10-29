# PDF 기반 LangChain RAG 시스템

Gemini 임베딩(`gemini-embedding-2`), 멀티모달 비전 OCR 및 Gemini LLM(`gemini-3.8-flash`), 그리고 별도 설치가 필요 없는 SQLite 기반 경량 벡터 스토어를 활용한 RAG(Retrieval-Augmented Generation) 파이프라인 및 시각화 대시보드입니다.

---

## 📁 폴더 구성

```text
rag/
├── rag_app.py              # FastAPI 기반 웹 대시보드 애플리케이션 (포트 8000)
├── rag_pipeline.py         # A1~A4 (인덱싱), B0~B4 (질의 확장 → 검색·리랭킹 → 답변) 전체 RAG 파이프라인
├── sqlite_vector_store.py  # SQLite + NumPy 기반 경량 벡터 스토어 (BLOB + 코사인 유사도 + BM25 RRF + 임베딩 캐시)
├── pricing_calculator.py   # 모델별 공식 단가표, 토큰·과금 누적 트래커
├── run_logger.py           # 인덱싱/질문 실행 로그(JSONL) 기록·조회·조합별 요약
├── gemini_error_debugger.py# Gemini API 오류(429 등) 진단
├── run_rag.py              # CLI 환경용 단계별 실행 스크립트
├── make_sample_pdf.py      # 테스트용 샘플 PDF 생성기
├── sample_knowledge.pdf    # 테스트용 기본 샘플 PDF 문서
├── han_gang_35.json        # 35문항 평가셋
├── capture_ui_35.py / eval_metrics_35.py / eval_refusal_35.py / build_report_35.py  # 성능 평가 파이프라인
├── RAG_성능평가_35문항.html  # 성능 평가 보고서
├── images/                 # 평가 화면 캡처 (q01~q35), early_ui/ 초기 UI 캡처
├── uploads/                # 업로드된 PDF (저장소 미포함) 및 캐시 파일
├── logs/                   # 실행 로그 index_runs.jsonl / query_runs.jsonl (저장소 미포함)
├── rag_sqlite.db           # 인덱싱된 청크 및 임베딩 벡터 저장 DB (저장소 미포함)
└── __init__.py             # Python 패키지 초기화 파일
```

## 🧩 주요 기능

- **검색 품질**: 질의 확장(동의어 키워드) + 하이브리드 검색(Dense + BM25, RRF) 후보 20개 → LLM 리랭킹으로 상위 `top_k`만 사용 (`RAGConfig.candidate_k`, `use_query_expansion`, `use_rerank`)
- **멀티 모델**: Gemini 3.8 Flash, Gemini 3.5 Flash-Lite, DeepSeek V4.1 Flash, Gemma 4 26B, Claude Haiku 5.5 (Vertex AI), Ollama gemma4:e2b
- **토큰 & 과금 대시보드**: 답변 생성뿐 아니라 질의 확장·리랭킹·임베딩·OCR 비용까지 모델별 집계
- **임베딩 캐시**: 헤더 배지 → 사용 ON/OFF, 저장 건수·용량, 현재 DB 청크 보유율, 적중률, 비우기
- **실행 로그**: '인덱싱 시작'/'질문하기' 1회마다 설정 조합과 결과(페이지·청크 수, 검색 결과, 토큰·비용, 소요 시간, 평가셋 질문이면 정답 페이지 적중·키워드 재현율)를 `logs/*.jsonl`에 기록. 헤더 **실행 로그** 버튼에서 조합별 비교 표로 조회·다운로드

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


---

## 📊 성능 평가 (35문항)

- 질문: `han_gang_35.json` (답변 가능 30 + 거절 평가 5)
- 보고서: `RAG_성능평가_35문항.html` (화면 캡처는 `images/q01~q35.png`)
- 실행 순서 (웹 서비스를 먼저 띄운 뒤): `capture_ui_35.py` → `eval_metrics_35.py` → `eval_refusal_35.py` → `build_report_35.py`
- 지표: Context Precision@4, Faithfulness, Negative Rejection Rate
- 페이지 단위 증분 인덱싱: 이미 적재된 PDF 쪽은 OCR·임베딩·저장을 건너뜁니다(`indexed_pages` 테이블, `content_hash` 검사).
- 원본 PDF(`uploads/*.pdf`)와 인덱싱된 DB, 청크 본문이 담긴 평가 원본(`ui_run_35.json`, `eval_*_35.json`)은 저작권·용량 문제로 저장소에 포함하지 않습니다. 평가를 재현하려면 같은 PDF를 업로드해 인덱싱하세요.
- 공개 보고서에는 검색 청크의 본문 없이 순위·쪽·점수만 표시합니다.
