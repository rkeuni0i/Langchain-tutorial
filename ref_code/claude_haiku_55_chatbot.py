"""
Google Cloud Vertex AI - Claude Haiku 5.5 (Anthropic)
LangChain 기반 챗봇 (CLI 대화형 + FastAPI 엔드포인트 지원)

Model ID: claude-haiku-5-5
연동 방식: Google Cloud Vertex AI (ChatAnthropicVertex / AnthropicVertex)
배포 리전: global
특징: 초고속 추론(High speed), 뛰어난 비용 효율성, 강력한 멀티턴 및 도구 호출 능력
참고: Claude Haiku 5.5는 모델 정책상 temperature 파라미터가 비활성화(deprecated)되어 기본 샘플링이 적용됩니다.
"""

import json
import os
import sys
from dotenv import load_dotenv

import google.auth
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_vertexai.model_garden import ChatAnthropicVertex

# Windows 터미널에서 한글 및 이모지 출력 시 cp949 인코딩 오류 방지
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# .env 파일 로드
load_dotenv(override=True)


# ==============================================================================
# [1] LangChain ChatAnthropicVertex 모델 팩토리 함수 및 유틸
# ==============================================================================
def extract_text_content(content) -> str:
    """AIMessage의 content(str 또는 블록 리스트)에서 텍스트 본문만 추출합니다."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)
        return "".join(text_parts)
    return str(content)


def get_haiku_55_model(
    project_id: str | None = None,
    location: str | None = None,
    max_output_tokens: int = 4096,
    streaming: bool = True,
    timeout: float = 120.0,
) -> ChatAnthropicVertex:
    """
    GCP Vertex AI의 claude-haiku-5-5 모델에 연결된
    LangChain ChatAnthropicVertex 인스턴스를 반환합니다.

    Args:
        project_id: GCP 프로젝트 ID (미지정 시 .env 또는 ADC 기본값 사용)
        location: 리전 (기본값: 'global')
        max_output_tokens: 최대 출력 토큰 수
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
    model_name = "claude-haiku-5-5"

    # 2. ChatAnthropicVertex 인스턴스 생성 (Claude Haiku 5.5는 temperature 미지정)
    chat_model = ChatAnthropicVertex(
        model_name=model_name,
        project=project,
        location=loc,
        credentials=credentials,
        max_output_tokens=max_output_tokens,
        streaming=streaming,
        timeout=timeout,
        max_retries=2,
    )

    # 유용한 메타데이터 바인딩
    object.__setattr__(chat_model, "gcp_project", project)
    object.__setattr__(chat_model, "gcp_location", loc)

    return chat_model


# ==============================================================================
# [2] CLI 대화형 챗봇 (터미널 모드)
# ==============================================================================
def run_cli_chatbot(
    system_prompt: str = (
        "당신은 Google Cloud Vertex AI에서 실행되는 Anthropic의 최신 고속 경량 모델 Claude Haiku 5.5입니다. "
        "사용자의 질문에 친절하고 정확하며 명확한 한국어로 성실히 답변해 주세요."
    ),
):
    """터미널에서 대화할 수 있는 인터랙티브 CLI 챗봇입니다."""
    print("=" * 75)
    print("⚡ Google Cloud Vertex AI - Claude Haiku 5.5 챗봇 시작")
    print("   (Model: claude-haiku-5-5 / Region: global)")
    print("=" * 75)

    try:
        model = get_haiku_55_model()
    except Exception as e:
        print(f"\n[초기화 오류] {e}\n")
        return

    project_id = getattr(model, "gcp_project", "N/A")
    location = getattr(model, "gcp_location", "N/A")

    print(f"📌 GCP 프로젝트 ID : {project_id}")
    print(f"📌 배포 리전        : {location}")
    print(f"📌 모델 식별자      : {model.model_name}")
    print("-" * 75)
    print("💡 사용 팁:")
    print("  - 대화 종료: 'exit', 'quit', 또는 'q' 입력")
    print("  - 대화 기록 초기화: '/clear' 또는 '/reset' 입력")
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

            if user_input.lower() in ("/info", "/model"):
                print(f"   [Model] {model.model_name}")
                print(f"   [Project] {project_id}")
                print(f"   [Location] {location}")
                print(f"   [History Length] {len(messages) - 1} messages")
                continue

            messages.append(HumanMessage(content=user_input))

            print("⚡ Claude Haiku 5.5: ", end="", flush=True)
            full_reply = ""

            for chunk in model.stream(messages):
                token = chunk.content
                if isinstance(token, str) and token:
                    print(token, end="", flush=True)
                    full_reply += token
                elif isinstance(token, list):
                    for part in token:
                        if isinstance(part, dict) and "text" in part:
                            print(part["text"], end="", flush=True)
                            full_reply += part["text"]

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
def test_single_turn(prompt: str = "Claude Haiku 5.5의 핵심 특징 3가지를 알려줘."):
    """단일 질문에 대해 응답을 테스트합니다."""
    print(f"\n[단일 질의 테스트] 질문: {prompt}")
    model = get_haiku_55_model(streaming=False)
    response = model.invoke([HumanMessage(content=prompt)])
    answer = extract_text_content(response.content)
    print(f"\n[Claude Haiku 5.5 답변]\n{answer}")
    return answer


@tool
def calculate_travel_budget(destination: str, days: int, hotel_per_night: int, daily_food: int) -> dict:
    """여행 목적지와 기간, 1박 숙박비 및 하루 식비를 입력받아 총 예상 경비를 계산합니다."""
    hotel_total = days * hotel_per_night
    food_total = days * daily_food
    total = hotel_total + food_total
    return {
        "destination": destination,
        "days": days,
        "hotel_total": hotel_total,
        "food_total": food_total,
        "total_cost": total,
    }


def test_tool_calling():
    """Claude Haiku 5.5의 LangChain 도구 호출 능력을 테스트합니다."""
    print("\n" + "=" * 75)
    print("🛠️ Claude Haiku 5.5 도구 호출(Tool Calling) 테스트")
    print("=" * 75)

    model = get_haiku_55_model(streaming=False)
    model_with_tools = model.bind_tools([calculate_travel_budget])

    question = "도쿄 4일 여행을 계획 중이야. 1박에 호텔비 12만원, 하루 식비 6만원이면 총 경비 계산해줘."
    print(f"질문: {question}")
    response = model_with_tools.invoke([HumanMessage(content=question)])
    print(f"Tool Calls: {response.tool_calls}")


# ==============================================================================
# [4] FastAPI 웹 API 서버
# ==============================================================================
api = FastAPI(
    title="Claude Haiku 5.5 Chat API",
    description="Google Cloud Vertex AI - Claude Haiku 5.5 LangChain FastAPI Server",
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
    max_output_tokens: int = 4096


@api.get("/health")
def health_check():
    return {"status": "ok", "model": "claude-haiku-5-5", "provider": "Google Cloud Vertex AI"}


@api.post("/api/chat")
def chat_completion(request: ChatRequest):
    """일반(Non-streaming) JSON 응답 엔드포인트"""
    try:
        model = get_haiku_55_model(
            max_output_tokens=request.max_output_tokens,
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
        content = extract_text_content(resp.content)
        return {"role": "assistant", "content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/chat/stream")
def chat_streaming(request: ChatRequest):
    """SSE(Server-Sent Events) 실시간 토큰 스트리밍 엔드포인트"""
    try:
        model = get_haiku_55_model(
            max_output_tokens=request.max_output_tokens,
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
                chunk_text = extract_text_content(chunk.content)
                if chunk_text:
                    yield f"data: {json.dumps({'content': chunk_text}, ensure_ascii=False)}\n\n"
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
        print("🚀 Claude Haiku 5.5 FastAPI 서버를 시작합니다: http://127.0.0.1:8003")
        uvicorn.run("claude_haiku_55_chatbot:api", host="127.0.0.1", port=8003, reload=True)
    elif "--single" in sys.argv:
        test_single_turn()
    elif "--tools" in sys.argv:
        test_tool_calling()
    else:
        run_cli_chatbot()
