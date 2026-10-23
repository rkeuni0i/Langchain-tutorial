import os
import sys
import google.auth
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# Windows 콘솔 출력 인코딩을 UTF-8로 설정 (특수문자 µg/m³, °C 등의 출력 에러 방지)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 1. 환경변수 로드 (.env 설정)
load_dotenv()

# 2. 내 컴퓨터의 ADC(Application Default Credentials) 자격증명 로드
# gcloud auth application-default login 으로 로그인된 계정 및 기본 프로젝트를 자동으로 가져옵니다.
try:
    credentials, project_id = google.auth.default()
except Exception as e:
    raise RuntimeError(
        f"ADC 인증 정보를 불러올 수 없습니다.\n"
        f"터미널에서 'gcloud auth application-default login'을 실행해 주세요.\n오류: {e}"
    )

project = os.getenv("GOOGLE_CLOUD_PROJECT") or project_id
location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")


# 3. 요구된 구조화된 출력 스키마 정의 (Pydantic BaseModel)
class WeatherOutput(BaseModel):
    """서울 송파구 문정동 기상 정보 스키마"""

    temperature: str = Field(
        description="현재 기온 (단위 포함, 예: '15°C')"
    )
    humidity: str = Field(
        description="현재 습도 (단위 포함, 예: '65%')"
    )
    precipitation: str = Field(
        description="현재 강수량 (단위 포함, 예: '0mm')"
    )
    fine_dust: str = Field(
        description="미세먼지 농도 및 등급 상태 (단위 포함, 예: '25 ㎍/㎥ (보통)' 또는 '좋음')"
    )


# 4. 제미나이(Gemini 2.5 Flash) 모델 설정 (ADC + Google Search 그라운딩 활성화)
# 별도의 외부 날씨 API 없이, 제미나이가 직접 최신 웹 정보를 실시간 검색하여 기상 데이터를 가져옵니다.
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    vertexai=True,
    credentials=credentials,
    project=project,
    location=location,
    model_kwargs={
        # 제미나이의 실시간 구글 검색(Google Search Grounding) 도구 연결
        "tools": [{"google_search": {}}]
    },
)

# 5. 구조화된 출력을 생성하는 에이전트 생성
agent = create_agent(
    model=model,
    tools=[],  # 제미나이 자체 구글 검색으로 실시간 정보를 수집하므로 추가 커스텀 도구 불필요
    response_format=WeatherOutput,
    system_prompt=(
        "너는 정확한 기상 및 환경 정보를 실시간 검색하여 제공하는 전문 AI 어시스턴트야. "
        "구글 검색 기능을 활용하여 사용자가 요청한 지역(서울 송파구 문정동 등)의 최신 기상 정보"
        "(기온, 습도, 강수량, 미세먼지)를 직접 확인하고, 각 항목의 단위와 상태를 명확히 포함하여 "
        "지정된 WeatherOutput 구조로 응답해."
    ),
)


def main():
    print(f"GCP 프로젝트: {project} (위치: {location})")
    print("제미나이가 실시간 검색을 통해 서울 송파구 문정동 기상 정보를 수집하는 중...\n")

    query = "서울 송파구 문정동의 최신 현재 기온, 습도, 강수량, 미세먼지 정보를 검색해서 알려줘."
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})

    # 구조화된 출력(Pydantic 객체) 추출
    weather: WeatherOutput = result["structured_response"]

    # 요구된 형식대로 출력
    print("=" * 30)
    print(f"기온 : {weather.temperature}")
    print(f"습도 : {weather.humidity}")
    print(f"강수량 : {weather.precipitation}")
    print(f"미세먼지 : {weather.fine_dust}")
    print("=" * 30)


if __name__ == "__main__":
    main()
