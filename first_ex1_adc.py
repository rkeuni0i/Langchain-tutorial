"""
API Key 없이 GCP ADC (gcloud auth application-default login) 인증만으로
Vertex AI Gemini 모델 및 LangChain 에이전트를 실행하는 예제 코드입니다.
"""

import os
from datetime import datetime

import google.auth
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. GCP ADC 인증 및 프로젝트 ID 설정 (API Key 불필요)
# 사전 준비: 터미널에서 `gcloud auth application-default login` 실행 완료 상태여야 합니다.
# ADC를 명시적으로 전달해 환경변수의 API 키와 혼용하지 않으며, gcloud 기본 프로젝트 ID도 함께 가져옵니다.
credentials, default_project = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip() or default_project
if not PROJECT_ID:
    raise ValueError(
        "Google Cloud 프로젝트 ID를 찾을 수 없습니다. "
        "`gcloud config set project 프로젝트ID`를 실행하거나 GOOGLE_CLOUD_PROJECT 환경변수를 설정하세요."
    )
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# 2. API Key 없이 Vertex AI Gemini 모델 초기화
# Gemini 3.8 실습에서는 low 추론 수준과 요청당 120초 제한을 사용합니다.
# 응답은 스트리밍 API로 받아 invoke가 최종 메시지로 모읍니다.
model = ChatGoogleGenerativeAI(
    model=MODEL_ID,
    vertexai=True,
    credentials=credentials,
    project=PROJECT_ID,
    location=LOCATION,
    reasoning_effort="low",
    streaming=True,
    timeout=120,
    max_retries=0,
)

# ------------------------------------------------------------------
# 테스트 1: 단일 LLM 호출 (제시해주신 예제)
# ------------------------------------------------------------------
print("=== [1. 단일 LLM 호출 (API Key 없음 / ADC 인증)] ===")
response = model.invoke("RAG의 원리를 한 문장으로 간단히 설명해줘.")
print(response.text)


# ------------------------------------------------------------------
# 테스트 2: Tool을 포함한 Agent 호출 (first_ex1.py와 동일한 기능)
# ------------------------------------------------------------------
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


def get_current_time() -> str:
    """Get the current local date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


agent = create_agent(
    model=model,
    tools=[get_weather, get_current_time],
    system_prompt="You are a helpful assistant",
)

print("\n=== [2. Agent 도구 호출 (API Key 없음 / ADC 인증)] ===")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco and what time is it now?"}]}
)
print(result["messages"][-1].content_blocks)

