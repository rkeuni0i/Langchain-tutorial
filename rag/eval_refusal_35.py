"""거절 능력(Negative Rejection / Noise Robustness) 채점 - han_gang_35.json 의 answerable=false 5문항

판정(LLM 판사, 검색 청크 전문 기준):
  refused          : 문서에 없다고 명확히 거절하고 지어낸 내용이 없음
  refused_with_extra: 거절하되 문서에 실제로 있는 관련 내용을 덧붙임(날조 없음)
  hallucinated     : 문서에 없는 내용을 사실처럼 답함
지표: Negative Rejection Rate = (refused + refused_with_extra) / 문항 수, Hallucination Rate
결과: rag/eval_refusal_35.json
"""
import json, re, sys, time
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
from rag_pipeline import PDFRAGPipeline, RAGConfig

OUT = CURRENT_DIR / "eval_refusal_35.json"


def judge_prompt(q, answer, chunks):
    ctx = "\n\n".join(f"[청크 {i+1}]\n{c['full_content']}" for i, c in enumerate(chunks))
    return f"""당신은 RAG 거절 능력 심사위원입니다. 이 질문은 참고 문서에 답이 없거나(문서 밖/거짓 전제/주제 무관) 답하면 안 되는 질문입니다.
생성 답변을 아래 셋 중 하나로 판정하세요.
- refused: 문서에 근거가 없다고 명확히 밝히고 답을 지어내지 않음
- refused_with_extra: 거절하되, 검색된 청크에 실제로 있는 관련 내용을 덧붙였고 그 내용이 청크와 일치함(날조 없음)
- hallucinated: 문서에 없는 내용을 사실처럼 답하거나, 거짓 전제를 받아들여 설명함

질문: {q}
생성 답변: {answer}

검색된 청크:
{ctx}

JSON만 출력: {{"verdict": "refused|refused_with_extra|hallucinated", "reason": "한 줄 근거"}}"""


def main():
    qs = [q for q in json.load(open(CURRENT_DIR / "han_gang_35.json", encoding="utf-8")) if q.get("answerable") is False]
    run = json.load(open(CURRENT_DIR / "ui_run_35.json", encoding="utf-8"))
    pipe = PDFRAGPipeline(config=RAGConfig(db_path=str(CURRENT_DIR / "rag_sqlite.db"), embedding_model="gemini-embedding-2",
                                           llm_model="gemini-3.8-flash", top_k=4, temperature=0.0))
    pipe.llm.temperature = 0.0
    res = {}
    for q in qs:
        k = str(q["id"]); ans = run[k]["response"]["answer"]
        vec = pipe._step_b1_embed_query(q["question"])["query_vector"]
        chunks = pipe._step_b2_search_chunks(vec, query_text=q["question"], top_k=4)["results_preview"]
        for _ in range(3):
            try:
                out = pipe.llm.invoke(judge_prompt(q["question"], ans, chunks)).content
                if isinstance(out, list):
                    out = "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in out)
                j = json.loads(re.search(r"\{.*\}", out, re.S).group(0)); break
            except Exception as e:
                print("retry", e); time.sleep(3)
        res[k] = {"id": q["id"], "type": q["refusal_type"], "verdict": j["verdict"], "reason": j.get("reason", ""),
                  "top_score": chunks[0]["score"] if chunks else None,
                  "chunks": [{"rank": c["rank"], "page": c["page"], "score": c["score"], "text": c["full_content"][:300]} for c in chunks]}
        print(q["id"], j["verdict"], j.get("reason", "")[:60], flush=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
