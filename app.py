"""
LangChain + GCP Vertex AI (ADC 인증) & 로컬 Ollama(gemma4:e2b) 하이브리드 웹 챗봇
(구글 실시간 검색 ON/OFF 토글 및 토큰/비용 카운터 포함)
실행 방법: uv run streamlit run app.py
"""

import os
import uuid
import streamlit as st
import google.auth
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver


# ==============================================================================
# [1] 모델 초기화 함수 정의 (캐싱 적용)
# ==============================================================================

@st.cache_resource
def get_vertex_model(enable_google_search: bool = False):
    """
    GCP ADC 인증을 거친 Vertex AI Gemini 모델 인스턴스를 반환합니다.
    - enable_google_search: True일 경우 제미나이의 실시간 구글 검색(Google Search Grounding) 활성화
    """
    credentials, default_project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip() or default_project
    if not project_id:
        raise ValueError(
            "GCP 프로젝트 ID를 찾을 수 없습니다. "
            "`gcloud config set project [프로젝트ID]`를 먼저 실행하세요."
        )

    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    model_id = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # 구글 검색(Google Search Grounding) 활성화 여부에 따른 model_kwargs 설정
    model_kwargs = {"tools": [{"google_search": {}}]} if enable_google_search else {}

    model = ChatGoogleGenerativeAI(
        model=model_id,
        vertexai=True,
        credentials=credentials,
        project=project_id,
        location=location,
        model_kwargs=model_kwargs,
        streaming=False,
        timeout=120,
        max_retries=3,
    )
    return model, project_id, location, model_id


@st.cache_resource
def get_ollama_model():
    """로컬 Ollama에서 실행되는 gemma4:e2b 모델 인스턴스를 반환합니다."""
    return ChatOllama(
        model="gemma4:e2b",
        temperature=0,
    )


def format_message_content(content) -> str:
    """Gemini 등 멀티모달 모델이 반환하는 content 블록 리스트에서 순수 텍스트만 추출합니다."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
            elif hasattr(part, "text"):
                parts.append(part.text)
        return "\n".join(parts) if parts else str(content)
    return str(content)


def calculate_token_usage_and_cost(response_messages, model_choice: str):
    """
    LangChain 응답 메시지에서 토큰 사용량을 추출하고 예상 비용을 계산합니다.
    """
    last_msg = response_messages[-1]
    input_tokens = 0
    output_tokens = 0

    # 1. usage_metadata 또는 response_metadata에서 토큰 수 추출
    usage = getattr(last_msg, "usage_metadata", None)
    if isinstance(usage, dict):
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
    elif hasattr(last_msg, "response_metadata") and isinstance(last_msg.response_metadata, dict):
        meta = last_msg.response_metadata
        if "prompt_eval_count" in meta or "eval_count" in meta:
            input_tokens = meta.get("prompt_eval_count", 0)
            output_tokens = meta.get("eval_count", 0)
        elif "usage_metadata" in meta and isinstance(meta["usage_metadata"], dict):
            input_tokens = meta["usage_metadata"].get("input_tokens", 0)
            output_tokens = meta["usage_metadata"].get("output_tokens", 0)

    # 2. 메타데이터에 토큰 정보가 없을 경우 보정 (한글/영문 글자 수 기반 추정)
    if input_tokens == 0 and output_tokens == 0:
        user_text = response_messages[-2].content if len(response_messages) >= 2 else ""
        assistant_text = format_message_content(last_msg.content)
        input_tokens = max(1, len(str(user_text)) // 3)
        output_tokens = max(1, len(assistant_text) // 3)

    total_tokens = input_tokens + output_tokens

    # 3. 비용 계산
    # Gemini 2.5 Flash 단가 기준:
    # - 입력: $0.075 / 1,000,000 토큰
    # - 출력: $0.30 / 1,000,000 토큰
    # - 환율: 1 USD ≈ 1,380 KRW
    if "Gemini" in model_choice:
        cost_usd = (input_tokens * 0.000000075) + (output_tokens * 0.00000030)
        cost_krw = cost_usd * 1380
    else:
        # 로컬 Ollama 모델은 로컬 GPU 연산이므로 API 비용 0원
        cost_usd = 0.0
        cost_krw = 0.0

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
        "cost_krw": cost_krw,
    }


# ==============================================================================
# [2] Streamlit 메인 앱 구성
# ==============================================================================
def main():
    st.set_page_config(page_title="AI 챗봇 (Gemini & Ollama)", page_icon="💬", layout="wide")
    st.title("💬 AI 하이브리드 챗봇")
    st.caption("Google Gemini (Vertex AI ADC) & 로컬 Ollama | 🌐 실시간 구글 검색 지원")

    # 세션 상태 및 InMemorySaver 초기화
    if "memory" not in st.session_state:
        st.session_state.memory = InMemorySaver()

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())[:8]

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "안녕하세요! 무엇이든 물어보세요. 사이드바에서 구글 검색 기능과 모델을 자유롭게 설정할 수 있습니다."}
        ]

    # 토큰 및 비용 통계 세션 상태
    if "token_stats" not in st.session_state:
        st.session_state.token_stats = {
            "last_input": 0,
            "last_output": 0,
            "total_input": 0,
            "total_output": 0,
            "total_cost_krw": 0.0,
            "total_cost_usd": 0.0,
        }

    # 사이드바: 모델 설정 및 구글 검색 ON/OFF
    with st.sidebar:
        st.header("⚙️ 모델 설정")
        model_choice = st.radio(
            "사용할 AI 모델 선택:",
            options=["Google Gemini (Vertex AI ADC)", "로컬 Ollama (gemma4:e2b)"],
            index=0,
            help="클라우드 Gemini 모델과 PC 로컬 Ollama 모델 중 원하는 모델을 선택하세요.",
        )

        st.divider()

        # ======================================================================
        # [신규 기능] 구글 검색(Google Search Grounding) ON/OFF 토글
        # ======================================================================
        st.subheader("🌐 구글 검색 기능")
        enable_search = st.toggle(
            "구글 실시간 검색 활성화",
            value=False,
            help="최신 뉴스, 날씨, 실시간 정보가 필요할 때 켭니다. (Google Gemini 지원)",
        )

        if enable_search:
            if "Gemini" in model_choice:
                st.caption("🟢 **구글 검색 ON**: 최신 웹 정보를 실시간 검색하여 답변합니다.")
            else:
                st.caption("⚠️ **알림**: 구글 검색 그라운딩은 Google Gemini 모델에서 지원됩니다. 상단에서 Gemini를 선택하시면 구글 검색이 적용됩니다.")
        else:
            st.caption("⚪ **구글 검색 OFF**: 모델의 자체 학습 지식만으로 답변합니다.")

        st.divider()

        # 선택된 설정에 따른 모델 로드
        try:
            if "Gemini" in model_choice:
                model, project_id, location, model_id = get_vertex_model(enable_google_search=enable_search)
                st.write(f"**활성 모델:** `Gemini ({model_id})`")
                st.write(f"**GCP 프로젝트:** `{project_id}`")
                st.write(f"**리전:** `{location}`")
                st.write(f"**검색 연동:** `{'ON (실시간 검색)' if enable_search else 'OFF'}`")
            else:
                model = get_ollama_model()
                st.write(f"**활성 모델:** `Ollama (gemma4:e2b)`")
                st.write(f"**엔드포인트:** `http://localhost:11434`")
                st.write(f"**실행 환경:** `로컬 GPU`")

            # 시스템 프롬프트 설정 (검색 활성화 여부에 따라 맞춤 안내)
            if enable_search and "Gemini" in model_choice:
                system_prompt = (
                    "당신은 친절하고 전문적인 AI 챗봇 어시스턴트입니다. "
                    "구글 검색 기능을 활용하여 최신 정보나 실시간 데이터(뉴스, 날씨, 시세 등)를 직접 검색하여 "
                    "정확하고 신뢰할 수 있는 정보를 한국어로 답변하세요. 사용자의 이전 대화 맥락도 잘 기억하세요."
                )
            else:
                system_prompt = (
                    "당신은 친절하고 전문적인 AI 챗봇 어시스턴트입니다. "
                    "사용자의 이전 대화 맥락을 잘 기억하여 자연스럽고 도움이 되는 한국어로 답변하세요."
                )

            # 선택된 모델과 세션 메모리로 에이전트 생성
            agent = create_agent(
                model=model,
                tools=[],
                checkpointer=st.session_state.memory,
                system_prompt=system_prompt,
            )
        except Exception as e:
            st.error(f"모델 초기화 오류: {e}")
            return

        st.divider()
        st.subheader("🧠 InMemorySaver 모니터링")
        st.write(f"**현재 세션 ID (thread_id):** `{st.session_state.thread_id}`")

        # InMemorySaver에 저장된 실제 메시지 조회
        current_config = {"configurable": {"thread_id": st.session_state.thread_id}}
        current_state = agent.get_state(current_config)

        saved_messages = current_state.values.get("messages", []) if current_state else []
        st.metric(label="InMemorySaver 누적 메시지 수", value=f"{len(saved_messages)}개")

        if saved_messages:
            with st.expander("저장된 메시지 구조 보기"):
                for idx, m in enumerate(saved_messages, 1):
                    msg_type = m.type if hasattr(m, "type") else "unknown"
                    clean_txt = format_message_content(m.content)
                    st.text(f"{idx}. [{msg_type.upper()}] {clean_txt[:25]}...")

        st.divider()
        if st.button("🗑️ 대화 초기화 (새 세션 시작)", use_container_width=True):
            st.session_state.thread_id = str(uuid.uuid4())[:8]
            st.session_state.chat_history = [
                {"role": "assistant", "content": "대화가 초기화되었습니다. 새로운 질문을 입력해 주세요!"}
            ]
            st.session_state.token_stats = {
                "last_input": 0,
                "last_output": 0,
                "total_input": 0,
                "total_output": 0,
                "total_cost_krw": 0.0,
                "total_cost_usd": 0.0,
            }
            st.rerun()

    # 기존 대화 히스토리 화면 출력
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(format_message_content(msg["content"]))

    # ==========================================================================
    # [3] 대화창 하단: 실시간 토큰 및 비용 카운터 대시보드
    # ==========================================================================
    st.markdown("---")
    stats = st.session_state.token_stats
    total_tokens = stats["total_input"] + stats["total_output"]

    counter_cols = st.columns([1, 1, 1.2, 1.5])
    with counter_cols[0]:
        st.metric("📥 이번 턴 입력", f"{stats['last_input']:,} 토큰")
    with counter_cols[1]:
        st.metric("📤 이번 턴 출력", f"{stats['last_output']:,} 토큰")
    with counter_cols[2]:
        st.metric("📊 세션 누적 토큰", f"{total_tokens:,} 토큰", help=f"누적 입력: {stats['total_input']:,} / 누적 출력: {stats['total_output']:,}")
    with counter_cols[3]:
        if "Ollama" in model_choice:
            st.metric("💰 누적 예상 비용", "0원 (로컬 무료)", help="로컬 Ollama 모델은 API 사용료가 발생하지 않습니다.")
        else:
            st.metric(
                "💰 누적 예상 비용",
                f"{stats['total_cost_krw']:.3f}원",
                delta=f"${stats['total_cost_usd']:.5f}",
                delta_color="off",
                help="Gemini 2.5 Flash 단가 기준: 입력 $0.075/1M, 출력 $0.30/1M (환율 1,380원/USD)",
            )

    # 사용자 입력 처리
    if user_input := st.chat_input("메시지를 입력하세요..."):
        # 1. 화면에 사용자 메시지 표시 및 기록
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # 2. 선택된 모델의 LangChain 에이전트 호출
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            active_model_label = "Gemini" if "Gemini" in model_choice else "Ollama (gemma4:e2b)"
            search_badge = " + 🌐구글 검색" if (enable_search and "Gemini" in model_choice) else ""
            with st.spinner(f"[{active_model_label}{search_badge}] 답변 생성 중..."):
                try:
                    config = {"configurable": {"thread_id": st.session_state.thread_id}}
                    response = agent.invoke(
                        {"messages": [{"role": "user", "content": user_input}]},
                        config=config,
                    )

                    # 최종 답변 메시지 추출 및 텍스트 포맷팅
                    raw_reply = response["messages"][-1].content
                    bot_reply = format_message_content(raw_reply)
                    st.write(bot_reply)

                    # 토큰 및 비용 통계 업데이트
                    turn_stats = calculate_token_usage_and_cost(response["messages"], model_choice)
                    st.session_state.token_stats["last_input"] = turn_stats["input_tokens"]
                    st.session_state.token_stats["last_output"] = turn_stats["output_tokens"]
                    st.session_state.token_stats["total_input"] += turn_stats["input_tokens"]
                    st.session_state.token_stats["total_output"] += turn_stats["output_tokens"]
                    st.session_state.token_stats["total_cost_krw"] += turn_stats["cost_krw"]
                    st.session_state.token_stats["total_cost_usd"] += turn_stats["cost_usd"]

                    # 세션 히스토리에 답변 추가 및 대시보드 갱신
                    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
                    st.rerun()

                except Exception as err:
                    st.error(f"답변 생성 중 오류가 발생했습니다: {err}")


if __name__ == "__main__":
    main()
