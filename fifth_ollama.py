from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

# 1. Ollama 모델 설정
model = ChatOllama(
    model="gemma4:e2b",
    temperature=0,
)

# 2. InMemorySaver 생성
memory = InMemorySaver()

# 3. 에이전트 생성
agent = create_agent(
    model=model,
    tools=[],
    checkpointer=memory,
    system_prompt="당신은 사용자의 대화를 잘 기억하고 친절하게 대답하는 AI 어시스턴트입니다.",
)


def print_divider(title: str):
    print("\n" + "=" * 60, flush=True)
    print(f" ▶ {title}", flush=True)
    print("=" * 60, flush=True)


def run_chat(user_msg: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    print(f"\n[세션 ID: {thread_id}] 사용자: {user_msg}", flush=True)

    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_msg}]},
        config=config,
    )
    bot_reply = result["messages"][-1].content
    print(f"[세션 ID: {thread_id}] AI 답변: {bot_reply}", flush=True)
    return result


if __name__ == "__main__":
    session_1 = "user_chulsoo"
    session_2 = "user_anonymous"

    # 테스트 1: 첫 번째 턴 (정보 주입)
    print_divider("테스트 1: session_1 사용자 정보 주입")
    run_chat("안녕! 내 이름은 철수고, 내가 가장 좋아하는 색깔은 파란색이야.", thread_id=session_1)

    # 테스트 2: 두 번째 턴 (동일 세션 기억 확인)
    print_divider("테스트 2: session_1 기억력 테스트 (동일 세션)")
    run_chat("내 이름이 뭔지, 그리고 내가 무슨 색을 좋아한다고 했는지 기억해?", thread_id=session_1)

    # 테스트 3: 세 번째 턴 (다른 세션 격리 확인)
    print_divider("테스트 3: session_2 새로운 사용자 테스트 (세션 격리 검증)")
    run_chat("내 이름과 좋아하는 색깔을 알고 있니?", thread_id=session_2)

    # 테스트 4: 저장된 체크포인트 데이터 확인
    print_divider("테스트 4: InMemorySaver 저장된 체크포인트 데이터 확인")
    state_session_1 = agent.get_state({"configurable": {"thread_id": session_1}})
    print(f"[{session_1}] 에 저장된 누적 메시지 개수: {len(state_session_1.values['messages'])}개")

    state_session_2 = agent.get_state({"configurable": {"thread_id": session_2}})
    print(f"[{session_2}] 에 저장된 누적 메시지 개수: {len(state_session_2.values['messages'])}개")
