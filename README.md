# LangChain Tutorial

LangChain 및 LangGraph 기초 튜토리얼 프로젝트입니다.

## 실습 파일 안내

### 1. Google Gemini (Vertex AI / GenAI) 버전
- **`first_ex1.py`**: Gemini 모델 및 도구(날씨, 현재 시각)를 사용하는 기본 에이전트 (API 키 방식)
- **`first_ex1_adc.py`**: API 키 없이 GCP ADC(`gcloud auth`) 인증으로 구동하는 Gemini 에이전트 (실시간 날씨 & 미세먼지)
- **`weather_structured_agent.py`**: Google Search Grounding 및 ADC 기반 실시간 기상 정보 구조화 에이전트
- **`third_ex.py`**: 웹페이지 텍스트 파싱 도구 및 딥 에이전트(`deepagents`) 실습
- **`fourth.py`**: 뉴스 기사 분석 및 Pydantic 기반 구조화된 출력(Structured Output) 에이전트
- **`fifth.py`**: `InMemorySaver`를 활용한 멀티턴 대화 세션 기억 및 세션 격리 테스트

### 2. 로컬 Ollama (`gemma4:e2b`) 버전
- **`first_ollama.py`** / **`second_ex.py`**: Ollama 로컬 모델 기반 도구 호출 에이전트
- **`third_ollama.py`**: Ollama 기반 웹 텍스트 파싱 및 딥 에이전트 실습
- **`fourth_ollama.py`**: Ollama 기반 뉴스 기사 구조화된 출력 에이전트
- **`fifth_ollama.py`**: Ollama 기반 `InMemorySaver` 대화 메모리 및 세션 격리 테스트

### 3. 웹 챗봇 서비스 (FastAPI)
- **`app.py`**: Gemini(ADC) & Ollama 하이브리드 웹 챗봇 (`InMemorySaver` 대화 기억 + 구글 실시간 검색 ON/OFF + 실시간 토큰/비용 카운터 탑재)
  ```powershell
  # 일반 python 명령어로 바로 실행 (브라우저: http://127.0.0.1:8000)
  uv run python app.py
  # 또는
  python app.py
  ```
  - 검색 모드(사이드바에서 선택): **OFF** / **Vertex AI Grounding with Google Search**(Gemini 전용) / **Custom Search**(`google_cse_search` 도구)
  - Custom Search 사용 시 `.env`에 `GOOGLE_CSE_API_KEY`, `GOOGLE_CSE_ID`(Programmable Search Engine의 cx) 설정 필요

---

## 환경 설정 및 실행 방법

1. **패키지 설치 및 가상환경 동기화 (`uv` 사용)**
   ```powershell
   uv sync
   ```

2. **환경 변수 설정 (Gemini 버전 실행 시)**
   - `.env.example` 파일을 복사하여 `.env` 파일을 생성합니다:
     ```env
     GEMINI_API_KEY=your_gemini_api_key_here
     ```

3. **스크립트 실행 예시**
   ```powershell
   # Gemini 버전 실행
   uv run python fifth.py

   # Ollama 버전 실행 (API 키 불필요)
   uv run python fifth_ollama.py
   ```
