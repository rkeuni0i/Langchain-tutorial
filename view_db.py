"""
PostgreSQL 데이터베이스에 저장된 챗봇 세션 및 대화 내역 조회 도구
실행: uv run python view_db.py
"""

import os
import sys
from dotenv import load_dotenv

# Windows 콘솔 한글 인코딩 보정
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()


def main():
    print("=" * 65)
    print("🔍 PostgreSQL 저장 데이터 조회기")
    print("=" * 65)

    if not DATABASE_URL:
        print("❌ .env 파일에 DATABASE_URL이 설정되어 있지 않습니다.")
        print("   예: DATABASE_URL=postgresql://postgres:postgres@localhost:5432/chatbot")
        print("=" * 65)
        return

    print(f"📌 연결 URL: {DATABASE_URL}\n")

    try:
        import psycopg
        from psycopg.rows import dict_row
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg_pool import ConnectionPool
    except ImportError as e:
        print(f"❌ 필수 라이브러리 누락: {e}")
        return

    # 1. DB 연결 테스트 및 테이블 목록 확인
    try:
        with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    ORDER BY table_name;
                    """
                )
                tables = [r["table_name"] for r in cur.fetchall()]

                print("📋 [1] 생성된 테이블 목록:")
                if not tables:
                    print("   (아직 생성된 테이블이 없습니다. app.py를 먼저 실행해 주세요.)")
                else:
                    for t in tables:
                        print(f"   • {t}")
                print()

                # 2. chat_sessions 테이블 조회
                if "chat_sessions" in tables:
                    cur.execute(
                        """
                        SELECT thread_id, COALESCE(title, '(제목 없음)') AS title, total_input, total_output, total_cost_krw, total_cost_usd, updated_at
                        FROM chat_sessions
                        ORDER BY updated_at DESC
                        LIMIT 10;
                        """
                    )
                    sessions = cur.fetchall()
                    print(f"📊 [2] chat_sessions 테이블 (최근 세션 {len(sessions)}개):")
                    if not sessions:
                        print("   (저장된 세션이 없습니다.)")
                    else:
                        for s in sessions:
                            total_tokens = s["total_input"] + s["total_output"]
                            print(f"   👉 세션 ID: {s['thread_id']} | 제목: {s['title']}")
                            print(f"      - 토큰: 입력 {s['total_input']:,} + 출력 {s['total_output']:,} = 총 {total_tokens:,} 토큰")
                            print(f"      - 누적 비용: {s['total_cost_krw']:.3f}원 (${s['total_cost_usd']:.5f})")
                            print(f"      - 최종 갱신: {s['updated_at']}")
                    print()

                # 3. checkpoints 테이블 로우 수 확인
                if "checkpoints" in tables:
                    cur.execute("SELECT count(*) AS cnt FROM checkpoints;")
                    cnt = cur.fetchone()["cnt"]
                    print(f"💾 [3] checkpoints 테이블 (누적 체크포인트: {cnt}개 스냅샷)")
                    print()

    except Exception as e:
        print(f"❌ PostgreSQL 연결 실패: {e}")
        print("\n💡 문제 해결 팁:")
        print("  1. PostgreSQL 서버가 켜져 있는지 확인하세요.")
        print("  2. Docker 사용 시: docker compose up -d")
        print("  3. .env 파일의 사용자명, 비밀번호, 포트가 맞는지 확인하세요.")
        print("=" * 65)
        return

    # 4. PostgresSaver를 통해 실제 대화 메시지 역직렬화 출력
    try:
        pool = ConnectionPool(
            conninfo=DATABASE_URL,
            min_size=1,
            max_size=2,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
            open=True,
        )
        saver = PostgresSaver(pool)
        
        # chat_sessions에 있는 thread들의 실제 대화 내용 읽기
        if tables and "chat_sessions" in tables and sessions:
            print("💬 [4] 세션별 저장된 실제 대화 메시지 (최근 세션 기준):")
            for s in sessions[:3]:
                t_id = s["thread_id"]
                config = {"configurable": {"thread_id": t_id}}
                tup = saver.get_tuple(config)
                raw_msgs = tup.checkpoint.get("channel_values", {}).get("messages", []) if tup else []
                
                print(f"\n   --- [세션: {t_id}] (총 {len(raw_msgs)}개 메시지) ---")
                if not raw_msgs:
                    print("   (메시지 없음)")
                for idx, m in enumerate(raw_msgs, 1):
                    role = getattr(m, "type", "unknown").upper()
                    content = getattr(m, "content", "")
                    if isinstance(content, list):
                        # 멀티모달 / 블록 형태 파싱
                        txt_list = [b["text"] for b in content if isinstance(b, dict) and "text" in b]
                        content = " ".join(txt_list)
                    content_preview = str(content).strip().replace("\n", " ")
                    if len(content_preview) > 60:
                        content_preview = content_preview[:60] + "..."
                    print(f"   {idx}. [{role}] {content_preview}")

        pool.close()
    except Exception as e:
        print(f"⚠️ 대화 메시지 역직렬화 중 오류: {e}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()

