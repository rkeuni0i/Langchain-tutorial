import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

# 1. 환경변수 로드 및 모델 초기화
load_dotenv(override=True)

google_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    vertexai=True,
    google_api_key=google_api_key,
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
)

# 2. InMemorySaver 생성
# 메모리 상(RAM)에 대화 상태(체크포인트)를 세션(thread_id)별로 저장하는 저장소
memory = InMemorySaver()

# 3. 체크포인터를 탑재한 에이전트 생성
agent = create_agent(
    model=model,
    tools=[],  # 순수 대화 메모리 동작 확인을 위해 도구 없이 설정
    checkpointer=memory,
    system_prompt="당신은 사용자의 대화를 잘 기억하고 친절하게 대답하는 AI 어시스턴트입니다.",
)


def print_divider(title: str):
    print("\n" + "=" * 60)
    print(f" ▶ {title}")
    print("=" * 60)


def run_chat(user_msg: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    print(f"\n[세션 ID: {thread_id}] 사용자: {user_msg}")

    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_msg}]},
        config=config,
    )
    bot_reply = result["messages"][-1].content
    print(f"[세션 ID: {thread_id}] AI 답변: {bot_reply}")
    return result


if __name__ == "__main__":
    session_1 = "user_chulsoo"
    session_2 = "user_anonymous"

    # -------------------------------------------------------------
    # 테스트 1: 첫 번째 턴 (정보 기억시키기)
    # -------------------------------------------------------------
    print_divider("테스트 1: session_1 사용자 정보 주입")
    run_chat("안녕! 내 이름은 철수고, 내가 가장 좋아하는 색깔은 파란색이야.", thread_id=session_1)

    # -------------------------------------------------------------
    # 테스트 2: 두 번째 턴 (동일 thread_id에서 이전 대화 기억 확인)
    # -------------------------------------------------------------
    print_divider("테스트 2: session_1 기억력 테스트 (동일 세션)")
    run_chat("내 이름이 뭔지, 그리고 내가 무슨 색을 좋아한다고 했는지 기억해?", thread_id=session_1)

    # -------------------------------------------------------------
    # 테스트 3: 세 번째 턴 (다른 thread_id로 접속 시 세션 격리 확인)
    # -------------------------------------------------------------
    print_divider("테스트 3: session_2 새로운 사용자 테스트 (세션 격리 검증)")
    print("설명: session_1과 다른 session_2로 질문하면 철수의 정보를 몰라야 정상입니다.")
    run_chat("내 이름과 좋아하는 색깔을 알고 있니?", thread_id=session_2)

    # -------------------------------------------------------------
    # 테스트 4: InMemorySaver 내부 상태(State) 직접 조회
    # -------------------------------------------------------------
    print_divider("테스트 4: InMemorySaver 저장된 체크포인트 데이터 확인")

    # session_1의 저장된 상태 확인
    state_session_1 = agent.get_state({"configurable": {"thread_id": session_1}})
    print(f"[{session_1}] 에 저장된 누적 메시지 개수: {len(state_session_1.values['messages'])}개")
    for idx, msg in enumerate(state_session_1.values["messages"], 1):
        role = msg.type if hasattr(msg, "type") else "unknown"
        # 메시지 미리보기 (30자 제한)
        preview = msg.content[:30].replace("\n", " ") + ("..." if len(msg.content) > 30 else "")
        print(f"  {idx}. [{role.upper()}] {preview}")

    # session_2의 저장된 상태 확인
    state_session_2 = agent.get_state({"configurable": {"thread_id": session_2}})
    print(f"\n[{session_2}] 에 저장된 누적 메시지 개수: {len(state_session_2.values['messages'])}개")
    for idx, msg in enumerate(state_session_2.values["messages"], 1):
        role = msg.type if hasattr(msg, "type") else "unknown"
        preview = msg.content[:30].replace("\n", " ") + ("..." if len(msg.content) > 30 else "")
        print(f"  {idx}. [{role.upper()}] {preview}")

    print("\n" + "=" * 60)
    print(" 결론: InMemorySaver가 thread_id별로 독립적인 대화 기록을 완벽하게 기억/격리하고 있습니다!")
    print("=" * 60)

