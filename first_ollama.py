datetime import datetime
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

# 1. 도구(Tools) 정의
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 2. 로컬 Ollama 모델 설정
model = ChatOllama(
    model="gemma4:e2b",
    temperature=0,
)

# 3. 에이전트 생성
agent = create_agent(
    model=model,
    tools=[get_weather, get_current_time],
    system_prompt="You are a helpful assistant",
)

# 4. 실행
if __name__ == "__main__":
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What's the weather in San Francisco and what time is it now?",
                }
            ]
        }
    )
    print(result["messages"][-1].content_blocks)

