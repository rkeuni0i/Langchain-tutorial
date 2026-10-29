"""
Google Cloud Vertex AI - Gemini 3.5 Flash-Lite
LangChain 기반 챗봇 (CLI 대화형 + FastAPI 엔드포인트 지원)

Model ID: gemini-3.5-flash-lite
특징: 초고속 추론(Low-latency), 고처리량(High-throughput), 1M Context Window, 경량 에이전트 작업에 최적화
"""

import json
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

import google.auth
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

# Windows 터미널에서 한글 및 이모지 출력 시 cp949 인코딩 오류 방지
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# .env 파일 로드
load_dotenv(override=True)


# ==============================================================================
# [1] LangChain ChatGoogleGenerativeAI 모델 팩토리 함수
# ==============================================================================
def get_gemini_35_flash_lite_model(
    project_id: str | None = None,
    location: str | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = 4096,
    reasoning_effort: str = "minimal",
    streaming: bool = True,
    timeout: float = 120.0,
) -> ChatGoogleGenerativeAI:
    """
    GCP Vertex AI의 gemini-3.5-flash-lite 모델에 연결된
    LangChain ChatGoogleGenerativeAI 인스턴스를 반환합니다.

    Args:
        project_id: GCP 프로젝트 ID (미지정 시 .env 또는 ADC 기본값 사용)
        location: 리전 (기본값: 'global')
        temperature: 샘플링 온도 (기본값 None - 모델 기본 샘플링 값 적용)
        max_output_tokens: 최대 출력 토큰 수
        reasoning_effort: 추론 수준 ('minimal', 'low', 'medium', 'high')
        streaming: 실시간 스트리밍 여부
        timeout: 요청 타임아웃(초)
    """
    # 1. GCP ADC 인증 및 프로젝트 ID 확인
    try:
        credentials, default_project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
    except Exception as e:
        raise RuntimeError(
            "GCP 인증 정보를 찾을 수 없습니다.\n"
            "터미널에서 'gcloud auth application-default login'을 실행해 주세요."
        ) from e

    project = project_id or os.getenv("GOOGLE_CLOUD_PROJECT", "").strip() or default_project
    if not project:
        raise ValueError(
            "GCP 프로젝트 ID가 지정되지 않았습니다. .env의 GOOGLE_CLOUD_PROJECT를 설정하세요."
        )

    loc = location or os.getenv("GOOGLE_CLOUD_LOCATION", "global").strip()
    model_name = "gemini-3.5-flash-lite"

    # 2. ChatGoogleGenerativeAI 인스턴스 생성 (Vertex AI 연동)
    init_kwargs = {
        "model": model_name,
        "vertexai": True,
        "credentials": credentials,
        "project": project,
        "location": loc,
        "max_output_tokens": max_output_tokens,
        "reasoning_effort": reasoning_effort,
        "streaming": streaming,
        "timeout": timeout,
        "max_retries": 2,
    }
    if temperature is not None:
        init_kwargs["temperature"] = temperature

    chat_model = ChatGoogleGenerativeAI(**init_kwargs)

    # 유용한 메타데이터 바인딩
    object.__setattr__(chat_model, "gcp_project", project)
    object.__setattr__(chat_model, "gcp_location", loc)
    object.__setattr__(chat_model, "reasoning_effort", reasoning_effort)

    return chat_model

    return chat_model


# ==============================================================================
# [2] CLI 대화형 챗봇 (터미널 모드)
# ==============================================================================
def run_cli_chatbot(
    system_prompt: str = (
        "당신은 Google Cloud Vertex AI의 최신 경량 고속 모델인 Gemini 3.5 Flash-Lite입니다. "
        "사용자의 질문에 대해 신속하고 명확하며 핵심을 찌르는 유익한 한국어로 답변해 주세요."
    ),
):
    """터미널에서 대화할 수 있는 인터랙티브 CLI 챗봇입니다."""
    print("=" * 75)
    print("⚡ Google Cloud Vertex AI - Gemini 3.5 Flash-Lite 챗봇 시작")
    print("   (Model: gemini-3.5-flash-lite / Region: global)")
    print("=" * 75)

    reasoning_effort = "minimal"
    try:
        model = get_gemini_35_flash_lite_model(reasoning_effort=reasoning_effort)
    except Exception as e:
        print(f"\n[초기화 오류] {e}\n")
        return

    project_id = getattr(model, "gcp_project", "N/A")
    location = getattr(model, "gcp_location", "N/A")

    print(f"📌 GCP 프로젝트 ID : {project_id}")
    print(f"📌 배포 리전        : {location}")
    print(f"📌 모델 식별자      : {model.model}")
    print(f"📌 추론 수준        : {reasoning_effort}")
    print("-" * 75)
    print("💡 사용 팁:")
    print("  - 대화 종료: 'exit', 'quit', 또는 'q' 입력")
    print("  - 대화 기록 초기화: '/clear' 또는 '/reset' 입력")
    print("  - 추론 수준 변경: '/reasoning minimal|low|medium|high'")
    print("  - 시스템 상태 확인: '/info'")
    print("-" * 75)

    messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]

    while True:
        try:
            user_input = input("\n👤 사용자: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print("👋 챗봇을 종료합니다.")
                break

            if user_input.lower() in ("/clear", "/reset"):
                messages = [SystemMessage(content=system_prompt)]
                print("🧹 대화 기록이 초기화되었습니다.")
                continue

            if user_input.lower().startswith("/reasoning"):
                parts = user_input.split()
                if len(parts) > 1 and parts[1].lower() in ("minimal", "low", "medium", "high"):
                    reasoning_effort = parts[1].lower()
                    model = get_gemini_35_flash_lite_model(reasoning_effort=reasoning_effort)
                    print(f"⚙️ 추론 수준(Reasoning Effort)이 '{reasoning_effort}'로 변경되었습니다.")
                else:
                    print("⚠️ 올바른 옵션: minimal, low, medium, high")
                continue

            if user_input.lower() in ("/info", "/model"):
                print(f"   [Model] {model.model}")
                print(f"   [Project] {project_id}")
                print(f"   [Location] {location}")
                print(f"   [Reasoning] {reasoning_effort}")
                print(f"   [History Length] {len(messages) - 1} messages")
                continue

            messages.append(HumanMessage(content=user_input))

            print("⚡ Gemini 3.5 Flash-Lite: ", end="", flush=True)
            full_reply = ""

            for chunk in model.stream(messages):
                token = chunk.text
                if token:
                    print(token, end="", flush=True)
                    full_reply += token

            print()
            messages.append(AIMessage(content=full_reply))

        except KeyboardInterrupt:
            print("\n👋 챗봇을 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ [호출 오류]: {e}")


# ==============================================================================
# [3] 단일 테스트 및 도구 호출(Tool Calling)
# ==============================================================================
def test_single_turn(prompt: str = "Gemini 3.5 Flash-Lite의 핵심 특징 3가지를 알려줘."):
    """단일 질문에 대해 응답을 테스트합니다."""
    print(f"\n[단일 질의 테스트] 질문: {prompt}")
    model = get_gemini_35_flash_lite_model(streaming=False)
    response = model.invoke([HumanMessage(content=prompt)])
    answer = response.text if hasattr(response, "text") and response.text else response.content
    print(f"\n[Gemini 3.5 Flash-Lite 답변]\n{answer}")
    return answer


@tool
def calculate_travel_cost(days: int, hotel_per_night: int, daily_budget: int) -> int:
    """여행 총 경비를 계산합니다."""
    return (days * hotel_per_night) + (days * daily_budget)


def test_tool_calling():
    """Gemini 3.5 Flash-Lite의 LangChain 도구 호출 능력을 테스트합니다."""
    print("\n" + "=" * 75)
    print("🛠️ Gemini 3.5 Flash-Lite 도구 호출(Tool Calling) 테스트")
    print("=" * 75)

    model = get_gemini_35_flash_lite_model(streaming=False)
    model_with_tools = model.bind_tools([calculate_travel_cost])

    question = "4일 여행이고 1박에 15만원, 하루 식비/교통비가 8만원이면 총 경비가 얼마인가요?"
    print(f"질문: {question}")
    response = model_with_tools.invoke([HumanMessage(content=question)])
    print(f"Tool Calls: {response.tool_calls}")


# ==============================================================================
# [4] FastAPI 웹 API 서버
# ==============================================================================
api = FastAPI(
    title="Gemini 3.5 Flash-Lite Chat API",
    description="GCP Vertex AI Gemini 3.5 Flash-Lite LangChain FastAPI Server",
    version="1.0.0",
)

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str = Field(..., description="user, assistant, 또는 system")
    content: str = Field(..., description="메시지 내용")


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    temperature: float | None = None
    max_output_tokens: int = 4096
    reasoning_effort: str = "minimal"


@api.get("/health")
def health_check():
    return {"status": "ok", "model": "gemini-3.5-flash-lite"}


@api.post("/api/chat")
def chat_completion(request: ChatRequest):
    """일반(Non-streaming) JSON 응답 엔드포인트"""
    try:
        model = get_gemini_35_flash_lite_model(
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
            reasoning_effort=request.reasoning_effort,
            streaming=False,
        )
        langchain_msgs: list[BaseMessage] = []
        for m in request.messages:
            if m.role == "user":
                langchain_msgs.append(HumanMessage(content=m.content))
            elif m.role == "assistant":
                langchain_msgs.append(AIMessage(content=m.content))
            elif m.role == "system":
                langchain_msgs.append(SystemMessage(content=m.content))

        resp = model.invoke(langchain_msgs)
        content = resp.text if hasattr(resp, "text") and resp.text else str(resp.content)
        return {"role": "assistant", "content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/chat/stream")
def chat_streaming(request: ChatRequest):
    """SSE(Server-Sent Events) 실시간 토큰 스트리밍 엔드포인트"""
    try:
        model = get_gemini_35_flash_lite_model(
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
            reasoning_effort=request.reasoning_effort,
            streaming=True,
        )
        langchain_msgs: list[BaseMessage] = []
        for m in request.messages:
            if m.role == "user":
                langchain_msgs.append(HumanMessage(content=m.content))
            elif m.role == "assistant":
                langchain_msgs.append(AIMessage(content=m.content))
            elif m.role == "system":
                langchain_msgs.append(SystemMessage(content=m.content))

        def event_generator():
            for chunk in model.stream(langchain_msgs):
                text_chunk = chunk.text
                if text_chunk:
                    yield f"data: {json.dumps({'content': text_chunk}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# [5] 메인 실행 진입점
# ==============================================================================
if __name__ == "__main__":
    if "--server" in sys.argv:
        import uvicorn
        print("🚀 FastAPI 서버를 시작합니다: http://127.0.0.1:8002")
        uvicorn.run("gemini_35_flash_lite_chatbot:api", host="127.0.0.1", port=8002, reload=True)
    elif "--single" in sys.argv:
        test_single_turn()
    elif "--tools" in sys.argv:
        test_tool_calling()
    else:
        run_cli_chatbot()
