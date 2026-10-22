# LangChain Tutorial

LangChain 및 LangGraph 기초 튜토리얼 프로젝트입니다.

## 실습 파일 안내

- **`main.py`**: LangChain 패키지 설치 및 버전 확인
- **`first_ex1.py`**: Google GenAI 및 도구(날씨, 현재 시각)를 사용하는 기본 에이전트
- **`second_ex.py`**: 로컬 Ollama 모델(`ChatOllama`) 연동 에이전트
- **`third_ex.py`**: 웹페이지 텍스트 파싱 도구 및 딥 에이전트(`deepagents`) 실습
- **`fourth.py`**: 뉴스 기사 분석 및 Pydantic 기반 구조화된 출력(Structured Output) 에이전트
- **`fifth.py`**: `InMemorySaver`를 활용한 멀티턴 대화 세션 기억 및 세션 격리 테스트

## 환경 설정 및 실행 방법

1. **패키지 설치 및 가상환경 동기화 (`uv` 사용)**
   ```powershell
   uv sync
   ```

2. **환경 변수 설정**
   - `.env.example` 파일을 복사하여 `.env` 파일을 생성합니다.
   - 본인의 Gemini API 키를 입력합니다:
     ```env
     GEMINI_API_KEY=your_gemini_api_key_here
     ```

3. **스크립트 실행 예시**
   ```powershell
   uv run python fifth.py
   ```

