"""UI 실행 결과(ui_run_35.json)를 RAGAS 방식 2개 지표로 채점한다.

- Context Precision@K : 검색된 청크 중 정답 도출에 유용한 청크가 상위에 얼마나 몰려 있는가 (LLM 판사, 참조 정답 기준)
- Faithfulness        : 생성 답변의 주장(claim) 중 검색 청크로 뒷받침되는 비율 (LLM 판사)

결과: rag/eval_metrics_35.json
"""

import json
import re
import sys
import time
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from rag_pipeline import PDFRAGPipeline, RAGConfig

TOP_K = 4
OUT = CURRENT_DIR / "eval_metrics_35.json"


def parse_json(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0))


def llm_json(llm, prompt: str, retries: int = 3):
    for i in range(retries):
        try:
            out = llm.invoke(prompt).content
            if isinstance(out, list):
                out = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in out)
            return parse_json(out)
        except Exception as e:
            print(f"   (재시도 {i + 1}: {str(e)[:80]})")
            time.sleep(3)
    raise RuntimeError("LLM 판정 실패")


def context_precision_prompt(question, reference, chunks):
    ctx = "\n\n".join(f"[청크 {i + 1}]\n{c['full_content']}" for i, c in enumerate(chunks))
    return f"""당신은 RAG 검색 품질 심사위원입니다. 질문과 참조 정답을 보고, 각 청크가 참조 정답의 핵심 사실을 도출하는 데 '직접 유용한지' 판정하세요.
- useful=1: 청크에 참조 정답의 핵심 사실(수치, 명칭, 인과 등)이 하나 이상 실질적으로 담겨 있음
- useful=0: 주제만 비슷하거나 무관함

질문: {question}
참조 정답: {reference}

{ctx}

JSON만 출력: {{"relevance": [청크1 판정, 청크2 판정, ...], "reason": "한 줄 요약"}}  (relevance 길이는 {len(chunks)})"""


def faithfulness_prompt(question, answer, chunks):
    ctx = "\n\n".join(f"[청크 {i + 1}]\n{c['full_content']}" for i, c in enumerate(chunks))
    return f"""당신은 RAG 환각 탐지 심사위원입니다.
1) 아래 '생성 답변'을 질문에 대한 개별 사실 주장(claim)들로 분해하세요. (출처 표기, 서두 문구는 제외)
2) 각 claim이 '검색된 청크 내용만으로' 뒷받침(추론 포함)되면 supported=1, 청크에 없거나 모순되면 0으로 판정하세요.

질문: {question}

생성 답변:
{answer}

검색된 청크:
{ctx}

JSON만 출력: {{"claims": [{{"claim": "...", "supported": 1 또는 0}}, ...], "unsupported_note": "근거 없는 주장 요약(없으면 빈 문자열)"}}"""


def average_precision(rel):
    hits, score = 0, 0.0
    for k, r in enumerate(rel, 1):
        if r:
            hits += 1
            score += hits / k
    return score / hits if hits else 0.0


def main():
    questions = {q["id"]: q for q in json.loads((CURRENT_DIR / "han_gang_35.json").read_text(encoding="utf-8"))}
    run = json.loads((CURRENT_DIR / "ui_run_35.json").read_text(encoding="utf-8"))
    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}

    pipeline = PDFRAGPipeline(config=RAGConfig(
        db_path=str(CURRENT_DIR / "rag_sqlite.db"),
        embedding_model="gemini-embedding-2",
        llm_model="gemini-3.8-flash",
        top_k=TOP_K,
        temperature=0.0,
    ))
    pipeline.llm.temperature = 0.0

    for qid in sorted(questions):
        if str(qid) in results:
            continue
        q = questions[qid]
        if q.get("answerable") is False:
            continue  # 거절 평가 문항은 eval_refusal_35.py에서 채점
        resp = run[str(qid)]["response"]
        answer = resp["answer"]
        print(f"[Q{qid:02d}] 채점 중...", flush=True)

        # UI와 동일 질의 → 동일 top_k 검색 (전체 청크 본문 확보)
        vec = pipeline._step_b1_embed_query(q["question"])["query_vector"]
        chunks = pipeline._step_b2_search_chunks(vec, query_text=q["question"], top_k=TOP_K)["results_preview"]

        cp = llm_json(pipeline.llm, context_precision_prompt(q["question"], q["expected_answer"], chunks))
        rel = [1 if int(x) else 0 for x in cp["relevance"]][: len(chunks)]
        ff = llm_json(pipeline.llm, faithfulness_prompt(q["question"], answer, chunks))
        claims = ff["claims"]
        sup = sum(1 for c in claims if int(c["supported"]))
        faith = sup / len(claims) if claims else 0.0

        results[str(qid)] = {
            "id": qid,
            "relevance": rel,
            "context_precision": round(average_precision(rel), 4),
            "hit": int(any(rel)),
            "cp_reason": cp.get("reason", ""),
            "claims": claims,
            "faithfulness": round(faith, 4),
            "unsupported_note": ff.get("unsupported_note", ""),
            "chunks": [{"rank": c["rank"], "page": c["page"], "chunk_index": c["chunk_index"],
                        "score": c["score"], "text": c["full_content"]} for c in chunks],
        }
        OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"   CP={results[str(qid)]['context_precision']:.2f} rel={rel} Faith={faith:.2f} ({sup}/{len(claims)})", flush=True)
        time.sleep(1.0)


if __name__ == "__main__":
    main()
