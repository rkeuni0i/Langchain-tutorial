import ssl
import urllib.request
from html.parser import HTMLParser
from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

# 1. 구조화된 출력 스키마 정의 (Pydantic BaseModel)
class NewsAnalysisResult(BaseModel):
    category: str = Field(description="뉴스의 분야/카테고리 (예: 정치, 경제, 사회 등)")
    title: str = Field(description="뉴스 기사의 정확한 제목")


# 2. 뉴스 기사 내용을 가져오는 Tool 정의
@tool
def fetch_news_content(url: str) -> str:
    """주어진 뉴스 기사 URL에 접속하여 기사의 텍스트 내용을 가져옵니다."""
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30, context=ctx) as response:
            html = response.read().decode("utf-8", errors="replace")
    except Exception as e:
        return f"기사 로딩 실패: {e}"

    class TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.texts = []
            self.skip = False

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style", "noscript", "header", "footer"):
                self.skip = True

        def handle_endtag(self, tag):
            if tag in ("script", "style", "noscript", "header", "footer"):
                self.skip = False

        def handle_data(self, data):
            if not self.skip:
                text = data.strip()
                if text:
                    self.texts.append(text)

    parser = TextExtractor()
    parser.feed(html)
    return "\n".join(parser.texts[:300])


# 3. 로컬 Ollama 모델 설정
model = ChatOllama(
    model="gemma4:e2b",
    temperature=0,
)

# 4. response_format 인자를 활용한 구조화된 출력 에이전트 생성
agent = create_agent(
    model=model,
    tools=[fetch_news_content],
    response_format=NewsAnalysisResult,
    system_prompt="You are a helpful assistant that analyzes news articles and returns structured information.",
)

# 5. 실행
if __name__ == "__main__":
    target_url = "https://www.hani.co.kr/arti/politics/politics_general/1283619.html"
    prompt = f"다음 뉴스 기사 링크의 카테고리와 제목을 분석해줘: {target_url}"

    print("Ollama 에이전트 실행 중...", flush=True)
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})

    structured_data: NewsAnalysisResult = result["structured_response"]
    print("\n--- 분석 결과 (구조화된 출력) ---")
    print(f"카테고리: {structured_data.category}")
    print(f"뉴스 제목: {structured_data.title}")
    print("\n--- JSON 덤프 ---")
    print(structured_data.model_dump_json(indent=2))

