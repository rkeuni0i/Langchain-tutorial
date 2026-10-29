"""
GCP Vertex AI Model Garden - DeepSeek V4.1 Flash API Service
LangChain 챗봇 테스트 스크립트

Model Garden 공식 명칭: DeepSeek V4.1 Flash API Service
MaaS Model ID: deepseek-ai/deepseek-v4.1-flash-maas
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Windows 터미널에서 이모지 및 다국어 출력이 깨지거나 인코딩 에러(cp949)가 나지 않도록 UTF-8 설정
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import google.auth
import google.auth.transport.requests
import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# .env 파일 로드
load_dotenv(override=True)


class GCPVertexAuth(httpx.Auth):
    """
    Google Cloud OAuth 인증 토큰을 자동으로 주입 및 갱신(Refresh)하는 httpx Auth 클래스입니다.
    GCP Access Token은 1시간(3,600초) 후 만료되므로, 요청 직전 토큰 유효성을 검사하여
    만료 시 자동으로 갱신함으로써 장시간 대화 시 발생하는 401 Unauthorized 오류를 방지합니다.
    """

    def __init__(self, credentials: google.auth.credentials.Credentials):
        self.credentials = credentials
        self._request = google.auth.transport.requests.Request()

    def auth_flow(self, request: httpx.Request):
        # 토큰이 없거나 만료되었을 경우 갱신
        if not self.credentials.valid:
            try:
                self.credentials.refresh(self._request)
            except Exception as e:
                raise RuntimeError(
                    "GCP 인증 토큰 갱신에 실패했습니다.\n"
                    "터미널에서 아래 명령어를 실행하여 다시 로그인해 주세요:\n"
                    "  gcloud auth application-default login\n"
                    f"상세 에러: {e}"
                ) from e

        request.headers["Authorization"] = f"Bearer {self.credentials.token}"
        yield request


def get_deepseek_chat_model(
    project_id: str | None = None,
    location: str | None = None,
    model_id: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = 2048,
    streaming: bool = True,
    timeout: float = 120.0,
) -> ChatOpenAI:
    """
    GCP Vertex AI Model Garden의 DeepSeek V4.1 Flash API Service에 연결된
    LangChain ChatOpenAI 인스턴스를 생성하여 반환합니다.

    Args:
        project_id: GCP 프로젝트 ID (미지정 시 환경변수 또는 gcloud 기본 프로젝트 사용)
        location: 배포 리전 ('global' 또는 'us-central1' 등, 기본값: 'global')
        model_id: 모델 ID (기본값: 'deepseek-ai/deepseek-v4.1-flash-maas')
        temperature: 샘플링 온도 (0.0 ~ 1.0)
        max_tokens: 최대 출력 토큰 수
        streaming: 응답 실시간 스트리밍 여부
        timeout: 요청 제한 시간(초)
    """
    # 1. GCP 인증 정보 및 프로젝트 ID 획득
    try:
        credentials, default_project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
    except Exception as e:
        raise RuntimeError(
            "GCP 인증 정보를 찾을 수 없습니다.\n"
            "터미널에서 'gcloud auth application-default login' 명령어를 실행하거나\n"
            "GOOGLE_APPLICATION_CREDENTIALS 환경 변수에 서비스 계정 키 파일 경로를 지정하세요."
        ) from e

    # 프로젝트 ID 결정
    project = project_id or os.getenv("GOOGLE_CLOUD_PROJECT", "").strip() or default_project
    if not project:
        raise ValueError(
            "GCP 프로젝트 ID가 지정되지 않았습니다.\n"
            "환경 변수 GOOGLE_CLOUD_PROJECT를 설정하거나 `gcloud config set project [PROJECT_ID]`를 실행하세요."
        )

    # 리전 및 모델 ID 결정 (DEEPSEEK_MODEL, DEEPSEEK_MODEL_ID 둘 다 지원)
    loc = location or os.getenv("GOOGLE_CLOUD_LOCATION", "global").strip()
    model = (
        model_id
        or os.getenv("DEEPSEEK_MODEL", "").strip()
        or os.getenv("DEEPSEEK_MODEL_ID", "").strip()
        or "deepseek-ai/deepseek-v4.1-flash-maas"
    )

    # 초기 토큰 갱신 테스트
    auth_request = google.auth.transport.requests.Request()
    try:
        if not credentials.valid:
            credentials.refresh(auth_request)
    except Exception as e:
        raise RuntimeError(
            "GCP 인증 토큰이 만료되었거나 갱신할 수 없습니다.\n"
            "터미널에서 다음 명령어를 실행하여 ADC 로그인을 다시 진행해 주세요:\n"
            "  gcloud auth application-default login\n"
            f"상세 에러: {e}"
        ) from e

    # 2. Vertex AI OpenAI 호환 엔드포인트 Base URL 구성 (GCP Model Garden 공식 가이드: /v1)
    if loc == "global":
        base_url = f"https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/endpoints/openapi"
    else:
        base_url = f"https://{loc}-aiplatform.googleapis.com/v1/projects/{project}/locations/{loc}/endpoints/openapi"

    # 3. 자동 토큰 갱신이 포함된 HTTP 클라이언트 생성
    http_client = httpx.Client(
        auth=GCPVertexAuth(credentials),
        timeout=timeout,
    )

    # 4. LangChain ChatOpenAI 객체 생성 (메타데이터 포함)
    chat_model = ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=credentials.token or "dummy-token-handled-by-auth",
        http_client=http_client,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
        metadata={
            "gcp_project": project,
            "gcp_location": loc,
            "gcp_base_url": base_url,
        },
    )

    # Pydantic v2 필드 유효성 검사를 우회하여 편의 속성 저장
    object.__setattr__(chat_model, "gcp_project", project)
    object.__setattr__(chat_model, "gcp_location", loc)
    object.__setattr__(chat_model, "gcp_base_url", base_url)

    return chat_model


def run_chatbot(
    system_prompt: str = (
        "당신은 Google Cloud Vertex AI의 DeepSeek V4.1 Flash 모델을 기반으로 동작하는 친절하고 유능한 AI 어시스턴트입니다. "
        "질문에 대해 명확하고 논리적이며 핵심을 짚는 답변을 제공하세요."
    ),
):
    """
    콘솔 환경에서 실시간 스트리밍 및 대화 기록(Memory)을 지원하는 대화형 챗봇을 실행합니다.
    """
    print("=" * 70)
    print("🚀 Google Cloud Vertex AI - DeepSeek V4.1 Flash 챗봇 시작")
    print("   (Model Garden 공식 명칭: DeepSeek V4.1 Flash API Service)")
    print("=" * 70)

    try:
        model = get_deepseek_chat_model()
    except Exception as e:
        print(f"\n[초기화 오류] {e}\n")
        return

    project_id = getattr(model, "gcp_project", None) or (model.metadata.get("gcp_project") if model.metadata else "N/A")
    location = getattr(model, "gcp_location", None) or (model.metadata.get("gcp_location") if model.metadata else "N/A")
    base_url = getattr(model, "gcp_base_url", None) or (model.metadata.get("gcp_base_url") if model.metadata else getattr(model, "openai_api_base", "N/A"))

    print(f"📌 GCP 프로젝트 ID : {project_id}")
    print(f"📌 배포 리전        : {location}")
    print(f"📌 모델 식별자      : {model.model_name}")
    print(f"📌 엔드포인트 URL   : {base_url}")
    print("-" * 70)
    print("💡 사용 팁:")
    print("  - 대화 종료: 'exit', 'quit', 또는 'q' 입력")
    print("  - 대화 기록 초기화: '/clear' 또는 '/reset' 입력")
    print("  - 시스템 설정 확인: '/info'")
    print("-" * 70)

    # 대화 히스토리 초기화 (시스템 프롬프트 추가)
    messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]

    while True:
        try:
            user_input = input("\n👤 사용자: ").strip()
            if not user_input:
                continue

            # 종료 명령어
            if user_input.lower() in ("exit", "quit", "q"):
                print("👋 챗봇을 종료합니다.")
                break

            # 초기화 명령어
            if user_input.lower() in ("/clear", "/reset"):
                messages = [SystemMessage(content=system_prompt)]
                print("🧹 대화 기록이 초기화되었습니다.")
                continue

            # 정보 확인 명령어
            if user_input.lower() in ("/info", "/model"):
                print(f"   [Model] {model.model_name}")
                print(f"   [Project] {project_id}")
                print(f"   [Location] {location}")
                print(f"   [Endpoint] {base_url}")
                print(f"   [History Length] {len(messages) - 1} messages")
                continue

            # 사용자 메시지 기록
            messages.append(HumanMessage(content=user_input))

            # DeepSeek 모델 스트리밍 호출
            print("🤖 DeepSeek V4.1 Flash: ", end="", flush=True)
            full_reply = ""

            for chunk in model.stream(messages):
                token = chunk.content
                if token:
                    print(token, end="", flush=True)
                    full_reply += token

            print()  # 줄바꿈

            # 어시스턴트 응답 기록
            messages.append(AIMessage(content=full_reply))

        except KeyboardInterrupt:
            print("\n👋 대화가 중단되었습니다. 챗봇을 종료합니다.")
            break
        except Exception as err:
            err_msg = str(err)
            print(f"\n❌ [호출 오류 발생]: {err_msg}")
            if "404" in err_msg or "NotFound" in err_msg:
                print("💡 [해결 가이드] GCP Vertex AI Model Garden에서 'DeepSeek V4.1 Flash API Service'를")
                print("   활성화(Enable)했는지 확인해 주세요.")
                print("   (MaaS 공식 모델 ID는 'deepseek-ai/deepseek-v4.1-flash-maas' 일 수 있습니다.)")
            elif "403" in err_msg or "PermissionDenied" in err_msg:
                print("💡 [해결 가이드] 해당 GCP 프로젝트에 Vertex AI 사용자(Vertex AI User) 권한이 있는지 확인하세요.")
            elif "401" in err_msg or "Reauthentication" in err_msg:
                print("💡 [해결 가이드] 'gcloud auth application-default login'을 실행해 인증을 갱신하세요.")


# ----------------------------------------------------------------------
# 단일 질의 테스트 함수
# ----------------------------------------------------------------------
def test_single_turn(prompt: str = "DeepSeek V4.1 Flash 모델의 핵심 강점을 3가지로 요약해줘."):
    """단일 질문에 대한 응답을 확인하는 테스트 함수입니다."""
    print(f"\n[단일 질의 테스트] 질문: {prompt}")
    model = get_deepseek_chat_model()
    response = model.invoke([HumanMessage(content=prompt)])
    print(f"[답변]\n{response.content}")
    return response.content


# ----------------------------------------------------------------------
# 도구(Tool Calling) 연동 테스트
# ----------------------------------------------------------------------
@tool
def get_current_datetime() -> str:
    """현재 날짜와 시간을 반환하는 도구입니다."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def compute_expression(expression: str) -> str:
    """사칙연산 등 수식을 계산합니다. 예: '128 * 45'"""
    try:
        # 안전한 간단 수식 평가
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "지원되지 않는 문자열이 포함되어 있습니다."
        return str(eval(expression, {"__builtins__": None}, {}))
    except Exception as e:
        return f"계산 실패: {e}"


def test_tool_calling():
    """DeepSeek V4.1 Flash의 함수/도구 호출(Tool Calling) 능력을 테스트합니다."""
    print("\n" + "=" * 70)
    print("🛠️ DeepSeek V4.1 Flash 도구 호출(Tool Calling) 테스트")
    print("=" * 70)

    model = get_deepseek_chat_model()
    model_with_tools = model.bind_tools([get_current_datetime, compute_expression])

    question = "지금 현재 시각이 몇 시이고, 125 * 38의 값은 얼마인가요?"
    print(f"질문: {question}")

    response = model_with_tools.invoke([HumanMessage(content=question)])
    print("\n[모델 응답 - Tool Calls 확인]:")
    print(f"내용: {response.content}")
    print(f"호출된 도구 목록: {response.tool_calls}")


if __name__ == "__main__":
    # 인자에 따라 모드 선택
    # python deepseek_v4_flash_chatbot.py -> 대화형 챗봇
    # python deepseek_v4_flash_chatbot.py --single -> 단일 질문 테스트
    # python deepseek_v4_flash_chatbot.py --tools -> 도구 호출 테스트
    if "--single" in sys.argv:
        test_single_turn()
    elif "--tools" in sys.argv:
        test_tool_calling()
    else:
        run_chatbot()

