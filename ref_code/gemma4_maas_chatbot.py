"""
GCP Vertex AI Model Garden - Gemma 4 26B MaaS API
LangChain 기반 챗봇 (CLI 대화형 + FastAPI 엔드포인트 지원)

참조: gemma4_ref.txt
- Model: google/gemma-4-26b-a4b-it-maas
- Endpoint: https://aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/global/endpoints/openapi
- Thinking Mode: chat_template_kwargs: {"enable_thinking": True}
- Max Tokens: up to 128,000
"""

import json
import os
import sys
from datetime import datetime
from typing import AsyncGenerator, Generator
from dotenv import load_dotenv

import google.auth
import google.auth.transport.requests
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Windows 터미널에서 한글 및 이모지 출력 시 cp949 인코딩 오류 방지
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# .env 파일 로드
load_dotenv(override=True)


# ==============================================================================
# [1] GCP OAuth 토큰 자동 갱신(Auto-refresh) HTTP 인증 핸들러
# ==============================================================================
class GCPVertexAuth(httpx.Auth):
    """
    Google Cloud OAuth 인증 토큰을 자동으로 주입 및 갱신(Refresh)하는 httpx Auth 클래스입니다.
    GCP Access Token(1시간 유효) 만료 시 자동 갱신하여 401 오류를 방지합니다.
    """

    def __init__(self, credentials: google.auth.credentials.Credentials):
        self.credentials = credentials
        self._request = google.auth.transport.requests.Request()

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, None, None]:
        if not self.credentials.valid:
            try:
                self.credentials.refresh(self._request)
            except Exception as e:
                raise RuntimeError(
                    "GCP 인증 토큰 갱신에 실패했습니다.\n"
                    "터미널에서 'gcloud auth application-default login' 명령어를 실행하세요.\n"
                    f"에러 내용: {e}"
                ) from e

        request.headers["Authorization"] = f"Bearer {self.credentials.token}"
        yield request


# ==============================================================================
# [2] LangChain ChatOpenAI 모델 팩토리 함수 (Gemma 4 26B MaaS)
# ==============================================================================
def get_gemma4_chat_model(
    project_id: str | None = None,
    location: str | None = None,
    model_id: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    enable_thinking: bool = True,
    streaming: bool = True,
    timeout: float = 180.0,
) -> ChatOpenAI:
    """
    gemma4_ref.txt 규격에 맞춰 GCP Vertex AI Model Garden의
    Gemma 4 26B MaaS 엔드포인트에 연결된 LangChain ChatOpenAI 인스턴스를 반환합니다.

    Args:
        project_id: GCP 프로젝트 ID (미지정 시 .env 또는 ADC 기본값 사용)
        location: 리전 (gemma4_ref.txt 기준: 'global')
        model_id: 모델명 (기본값: 'google/gemma-4-26b-a4b-it-maas')
        temperature: 샘플링 온도
        max_tokens: 최대 출력 토큰 수 (최대 128,000까지 지원)
        enable_thinking: 사고(추론) 프로세스 활성화 여부
        streaming: 실시간 토큰 스트리밍 여부
        timeout: 요청 타임아웃(초)
    """
    # 1. GCP 인증 정보 및 프로젝트 ID 확인
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
    model_name = (
        model_id
        or os.getenv("GEMMA4_MODEL", "").strip()
        or "google/gemma-4-26b-a4b-it-maas"
    )

    # 2. Base URL 생성 (gemma4_ref.txt: /v1/projects/.../locations/.../endpoints/openapi)
    if loc == "global":
        base_url = f"https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/endpoints/openapi"
    else:
        base_url = f"https://{loc}-aiplatform.googleapis.com/v1/projects/{project}/locations/{loc}/endpoints/openapi"

    # 3. HTTP 클라이언트 생성 (자동 토큰 갱신 적용)
    http_client = httpx.Client(
        auth=GCPVertexAuth(credentials),
        timeout=timeout,
    )

    # 4. extra_body에 thinking 옵션 구성 (gemma4_ref.txt 규격)
    extra_body = {
        "chat_template_kwargs": {
            "enable_thinking": enable_thinking
        }
    }

    # 5. LangChain ChatOpenAI 객체 생성
    chat_model = ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=credentials.token or "dummy-token-handled-by-auth",
        http_client=http_client,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
        extra_body=extra_body,
        metadata={
            "gcp_project": project,
            "gcp_location": loc,
            "gcp_base_url": base_url,
            "enable_thinking": enable_thinking,
        },
    )

    object.__setattr__(chat_model, "gcp_project", project)
    object.__setattr__(chat_model, "gcp_location", loc)
    object.__setattr__(chat_model, "gcp_base_url", base_url)
    object.__setattr__(chat_model, "enable_thinking", enable_thinking)

    return chat_model


# ==============================================================================
# [3] CLI 대화형 챗봇 (터미널 모드)
# ==============================================================================
def run_cli_chatbot(
    system_prompt: str = (
        "당신은 Google DeepMind가 개발하고 GCP Vertex AI Model Garden에서 제공되는 "
        "Gemma 4 (26B-A4B-IT) 모델 기반의 지능형 AI 비서입니다. "
        "질문에 대해 정확하고 친절하며 논리적인 한국어로 답변해 주세요."
    ),
):
    """터미널에서 대화할 수 있는 인터랙티브 CLI 챗봇입니다."""
    print("=" * 75)
    print("💎 GCP Vertex AI - Gemma 4 26B MaaS 챗봇 시작")
    print("   (참조 규격: google/gemma-4-26b-a4b-it-maas / thinking: ON)")
    print("=" * 75)

    enable_thinking = True
    try:
        model = get_gemma4_chat_model(enable_thinking=enable_thinking)
    except Exception as e:
        print(f"\n[초기화 오류] {e}\n")
        return

    project_id = getattr(model, "gcp_project", "N/A")
    location = getattr(model, "gcp_location", "N/A")
    base_url = getattr(model, "gcp_base_url", "N/A")

    print(f"📌 GCP 프로젝트 ID : {project_id}")
    print(f"📌 배포 리전        : {location}")
    print(f"📌 모델 식별자      : {model.model_name}")
    print(f"📌 추론(Thinking)   : {'활성화 (ON)' if enable_thinking else '비활성화 (OFF)'}")
    print(f"📌 엔드포인트 URL   : {base_url}")
    print("-" * 75)
    print("💡 사용 팁:")
    print("  - 대화 종료: 'exit', 'quit', 또는 'q' 입력")
    print("  - 대화 기록 초기화: '/clear' 또는 '/reset' 입력")
    print("  - 추론 토글: '/thinking on' 또는 '/thinking off'")
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

            if user_input.lower() == "/thinking on":
                enable_thinking = True
                model = get_gemma4_chat_model(enable_thinking=True)
                print("🧠 Gemma 4 심층 추론(Thinking) 모드가 활성화되었습니다.")
                continue

            if user_input.lower() == "/thinking off":
                enable_thinking = False
                model = get_gemma4_chat_model(enable_thinking=False)
                print("⚡ Gemma 4 빠른 응답(Thinking OFF) 모드로 전환되었습니다.")
                continue

            if user_input.lower() in ("/info", "/model"):
                print(f"   [Model] {model.model_name}")
                print(f"   [Project] {project_id}")
                print(f"   [Location] {location}")
                print(f"   [Thinking] {'ON' if enable_thinking else 'OFF'}")
                print(f"   [History Length] {len(messages) - 1} messages")
                continue

            messages.append(HumanMessage(content=user_input))

            print("🤖 Gemma 4: ", end="", flush=True)
            full_reply = ""

            for chunk in model.stream(messages):
                token = chunk.content
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
# [4] 단일 테스트 및 도구 호출(Tool Calling)
# ==============================================================================
def test_single_turn(prompt: str = "Summer travel plan to Paris with 3 highlights"):
    """gemma4_ref.txt에 기재된 예시 프롬프트로 단일 호출을 테스트합니다."""
    print(f"\n[단일 질의 테스트] 질문: {prompt}")
    model = get_gemma4_chat_model(enable_thinking=True, max_tokens=2048)
    response = model.invoke([HumanMessage(content=prompt)])
    print("\n[Gemma 4 답변]\n" + response.content)
    return response.content


@tool
def get_current_time() -> str:
    """현재 날짜와 시간을 반환합니다."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def test_tool_calling():
    """Gemma 4 26B MaaS의 LangChain 도구 호출 능력을 테스트합니다."""
    print("\n" + "=" * 75)
    print("🛠️ Gemma 4 도구 호출(Tool Calling) 테스트")
    print("=" * 75)

    model = get_gemma4_chat_model(enable_thinking=False)
    model_with_tools = model.bind_tools([get_current_time])

    question = "지금 현재 시간이 몇 시인가요?"
    print(f"질문: {question}")
    response = model_with_tools.invoke([HumanMessage(content=question)])
    print(f"내용: {response.content}")
    print(f"Tool Calls: {response.tool_calls}")


# ==============================================================================
# [5] FastAPI 웹 API 서버 (FastAPI 네이티브 통합)
# ==============================================================================
api = FastAPI(
    title="Gemma 4 26B MaaS Chat API",
    description="GCP Vertex AI Model Garden Gemma 4 (26B-A4B-IT) LangChain FastAPI Server",
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
    enable_thinking: bool = True
    temperature: float = 0.7
    max_tokens: int = 4096


@api.get("/health")
def health_check():
    return {"status": "ok", "model": "google/gemma-4-26b-a4b-it-maas"}


@api.post("/api/chat")
def chat_completion(request: ChatRequest):
    """일반(Non-streaming) JSON 응답 엔드포인트"""
    try:
        model = get_gemma4_chat_model(
            enable_thinking=request.enable_thinking,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
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
        return {"role": "assistant", "content": resp.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/chat/stream")
def chat_streaming(request: ChatRequest):
    """SSE(Server-Sent Events) 실시간 토큰 스트리밍 엔드포인트"""
    try:
        model = get_gemma4_chat_model(
            enable_thinking=request.enable_thinking,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
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
                if chunk.content:
                    yield f"data: {json.dumps({'content': chunk.content}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# [6] 메인 실행 진입점
# ==============================================================================
if __name__ == "__main__":
    if "--server" in sys.argv:
        import uvicorn
        print("🚀 FastAPI 서버를 시작합니다: http://127.0.0.1:8001")
        uvicorn.run("gemma4_maas_chatbot:api", host="127.0.0.1", port=8001, reload=True)
    elif "--single" in sys.argv:
        test_single_turn()
    elif "--tools" in sys.argv:
        test_tool_calling()
    else:
        run_cli_chatbot()

