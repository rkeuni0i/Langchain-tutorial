"""ui_run_35.json + eval_metrics_35.json + han_gang_35.json → RAG_성능평가_35문항.html"""

import html
import json
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
OUT = CURRENT_DIR / "RAG_성능평가_35문항.html"
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

esc = html.escape
# 공개 저장소용: 검색 청크는 본문(책 내용) 없이 순위·쪽·점수 메타데이터만 노출


questions = json.loads((CURRENT_DIR / "han_gang_35.json").read_text(encoding="utf-8"))
run = json.loads((CURRENT_DIR / "ui_run_35.json").read_text(encoding="utf-8"))
metrics = json.loads((CURRENT_DIR / "eval_metrics_35.json").read_text(encoding="utf-8"))

refusal = json.loads((CURRENT_DIR / "eval_refusal_35.json").read_text(encoding="utf-8"))
refusal_qs = [q for q in questions if q.get("answerable") is False]
rows = []
for q in [q for q in questions if q.get("answerable") is not False]:
    k = str(q["id"])
    m = metrics[k]
    rows.append({**q, **m, "answer": run[k]["response"]["answer"],
                 "latency": run[k]["response"]["step_b4_generation"]["latency"]})

n = len(rows)
avg_cp = sum(r["context_precision"] for r in rows) / n
avg_ff = sum(r["faithfulness"] for r in rows) / n
hit = sum(r["hit"] for r in rows)
top1 = sum(1 for r in rows if r["relevance"][0])
avg_lat = sum(r["latency"] for r in rows) / n
total_claims = sum(len(r["claims"]) for r in rows)
bad_claims = sum(1 for r in rows for c in r["claims"] if not int(c["supported"]))


def tone(v, good=0.9, warn=0.7):
    return "good" if v >= good else "warn" if v >= warn else "bad"


def grade(v):
    return "우수" if v >= 0.9 else "양호" if v >= 0.7 else "미흡"


def by_group(key):
    groups = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)
    out = []
    for g, rs in groups.items():
        out.append((g, len(rs), sum(x["context_precision"] for x in rs) / len(rs),
                    sum(x["faithfulness"] for x in rs) / len(rs)))
    return out


def bars(items):
    h = ""
    for label, cnt, cp, ff in items:
        h += f"""<tr><th>{esc(label)} <small>({cnt})</small></th>
<td><div class="bar"><i class="{tone(cp)}" style="width:{cp*100:.0f}%"></i><b>{cp:.2f}</b></div></td>
<td><div class="bar"><i class="{tone(ff)}" style="width:{ff*100:.0f}%"></i><b>{ff:.2f}</b></div></td></tr>"""
    return h


order = {"하": 0, "중": 1, "상": 2}
diff_items = sorted(by_group("difficulty"), key=lambda x: order[x[0]])
cat_items = sorted(by_group("category"), key=lambda x: x[0])
chap_items = by_group("chapter")

# 청크 위치 평균 (relevance 분포)
rank_rel = [sum(r["relevance"][i] for r in rows) for i in range(4)]

table = ""
for r in rows:
    rel = "".join(f'<span class="dot {"on" if x else ""}" title="rank {i+1}"></span>' for i, x in enumerate(r["relevance"]))
    table += f"""<tr onclick="location.hash='q{r['id']}'">
<td>Q{r['id']:02d}</td><td>{esc(r['chapter'])}</td><td>{esc(r['category'])}</td><td>{r['difficulty']}</td>
<td class="q">{esc(r['question'][:46])}…</td><td>{rel}</td>
<td class="{tone(r['context_precision'])}">{r['context_precision']:.2f}</td>
<td class="{tone(r['faithfulness'], 0.95, 0.8)}">{r['faithfulness']:.2f}</td></tr>"""

cards = ""
for r in rows:
    claims = "".join(
        f'<li class="{"ok" if int(c["supported"]) else "ng"}">{"✔" if int(c["supported"]) else "✘"} {esc(c["claim"])}</li>'
        for c in r["claims"])
    chunks = ""
    for i, c in enumerate(r["chunks"]):
        useful = r["relevance"][i] if i < len(r["relevance"]) else 0
        chunks += f"""<div class="chunk"><span class="tag {"on" if useful else ""}">{"유용" if useful else "무관"}</span>
#{c['rank']} · p.{c['page']} · 청크 {c['chunk_index']} · RRF {c['score']}</div>"""
    note = f'<p class="note">⚠ {esc(r["unsupported_note"])}</p>' if r["unsupported_note"] else ""
    cards += f"""<section class="card" id="q{r['id']}">
<header><h3>Q{r['id']:02d}. {esc(r['question'])}</h3>
<div class="meta">{esc(r['chapter'])} · {esc(r['category'])} · 난이도 {r['difficulty']} · 정답 위치 {esc(r['page_range'])} · 응답 {r['latency']}s</div>
<div class="scores"><span class="pill {tone(r['context_precision'])}">Context Precision {r['context_precision']:.2f}</span>
<span class="pill {tone(r['faithfulness'], 0.95, 0.8)}">Faithfulness {r['faithfulness']:.2f}</span></div></header>
<a href="images/q{r['id']:02d}.png" target="_blank"><img loading="lazy" src="images/q{r['id']:02d}.png" alt="Q{r['id']} 화면 캡처"></a>
<div class="cols"><div><h4>참조 정답</h4><p>{esc(r['expected_answer'])}</p>
<h4>검색 청크 (top-4)</h4>{chunks}<p class="note">{esc(r['cp_reason'])}</p></div>
<div><h4>Faithfulness 판정 (주장 {len(r['claims'])}개)</h4><ul class="claims">{claims}</ul>{note}</div></div>
</section>"""


# ---------------- 거절 능력 평가(35문항 중 5개) ----------------
TYPE_LABEL = {"unanswerable": "문서에 없는 사실", "false_premise": "거짓 전제", "noise": "노이즈(주제 무관)"}
VERDICT_LABEL = {"refused": "정상 거절", "refused_with_extra": "거절 + 관련 정보 덧붙임", "hallucinated": "환각 답변"}
n_ref = len(refusal_qs)
n_refused = sum(1 for v in refusal.values() if v["verdict"] in ("refused", "refused_with_extra"))
n_pure = sum(1 for v in refusal.values() if v["verdict"] == "refused")
n_hall = sum(1 for v in refusal.values() if v["verdict"] == "hallucinated")
ans_top1 = [metrics[str(r["id"])]["chunks"][0]["score"] for r in rows]
ref_top1 = [v["top_score"] for v in refusal.values()]

ref_table = ""
ref_cards = ""
for q in refusal_qs:
    k = str(q["id"])
    v = refusal[k]
    ans = run[k]["response"]["answer"]
    lat = run[k]["response"]["step_b4_generation"]["latency"]
    ok = v["verdict"] != "hallucinated"
    ref_table += (
        f"<tr onclick=\"location.hash='q{q['id']}'\"><td>Q{q['id']}</td><td>{TYPE_LABEL[v['type']]}</td>"
        f"<td class=\"q\">{esc(q['question'])}</td>"
        f"<td class=\"{'good' if ok else 'bad'}\"><b>{VERDICT_LABEL[v['verdict']]}</b></td>"
        f"<td>{v['top_score']:.2f}</td></tr>"
    )
    chunk_html = "".join(
        f"<div class=\"chunk\">#{c['rank']} · p.{c['page']} · RRF {c['score']}</div>"
        for c in v["chunks"]
    )
    ref_cards += f"""<section class="card" id="q{q['id']}">
<header><h3>Q{q['id']}. {esc(q['question'])}</h3>
<div class="meta">{TYPE_LABEL[v['type']]} · 응답 {lat}s · 검색 1순위 RRF {v['top_score']:.2f}</div>
<div class="scores"><span class="pill {'good' if ok else 'bad'}">{VERDICT_LABEL[v['verdict']]}</span></div></header>
<a href="images/q{q['id']:02d}.png" target="_blank"><img loading="lazy" src="images/q{q['id']:02d}.png" alt="Q{q['id']} 화면 캡처"></a>
<div class="cols"><div><h4>기대 동작</h4><p>{esc(q['expected_answer'])}</p><h4>판정 근거</h4><p>{esc(v['reason'])}</p></div>
<div><h4>검색된 청크 (모두 질문과 무관 또는 부분 관련)</h4>{chunk_html}</div></div>
</section>"""

refusal_section = f"""<h2>10. 거절 능력 평가 — Negative Rejection · 노이즈 (추가 5문항)</h2>
<p class="def">문서에 답이 없거나 답하면 안 되는 질문 5개(문서 밖 사실 2, 거짓 전제 1, 주제 무관 노이즈 2)를 han_gang_35.json에 추가하고(Q31~Q35), 앞의 30문항과 같은 방식으로 실제 웹 UI에 입력해 화면을 캡처했습니다(images/q31~q35.png). 판정은 LLM 판사가 검색 청크 전문을 근거로 <b>정상 거절 / 거절+관련 정보 덧붙임 / 환각 답변</b> 중 하나로 매겼습니다.</p>
<div class="grid">
<div class="kpi"><span>Negative Rejection Rate</span><b class="{'good' if n_refused == n_ref else 'warn'}">{n_refused}/{n_ref}</b><span>{n_refused / n_ref * 100:.0f}% · 지어내지 않고 거절</span></div>
<div class="kpi"><span>환각 답변</span><b class="{'good' if n_hall == 0 else 'bad'}">{n_hall}/{n_ref}</b><span>문서에 없는 내용을 사실처럼 답함</span></div>
<div class="kpi"><span>순수 거절 / 거절+부가정보</span><b>{n_pure} / {n_refused - n_pure}</b><span>부가정보는 문서 내용과 일치</span></div>
<div class="kpi"><span>검색 1순위 RRF 점수</span><b>{min(ref_top1):.2f}~{max(ref_top1):.2f}</b><span>답변 가능 문항은 {min(ans_top1):.2f}~{max(ans_top1):.2f}</span></div>
</div>
<div class="tbl" style="margin-top:12px"><table><tr><th>#</th><th>유형</th><th>질문</th><th>판정</th><th>1순위 RRF</th></tr>{ref_table}</table></div>
<div class="box" style="margin-top:12px"><b>해석.</b>
<ul>
<li><b>5문항 모두 환각 없이 거절했습니다.</b> 문서 밖 사실(Q31·Q32)과 주제 무관 질문(Q34·Q35)은 "참고자료에 없어 답할 수 없다"고 답했습니다.</li>
<li><b>거짓 전제(Q33)도 받아들이지 않았습니다.</b> "양화대교 1990년 붕괴"는 문서에 없다고 밝히고, 문서에 실제로 있는 내용(중차량 통과 후에도 붕괴하지 않았다는 점, 성수대교 붕괴 이후 1996년부터 보강·재시공)만 덧붙였습니다. 덧붙인 내용은 청크와 일치해 날조가 아닙니다.</li>
<li><b>거절은 검색이 아니라 프롬프트의 "자료에 없으면 답하지 말라"는 지시 덕분입니다.</b> 거절 문항에서도 검색은 무관한 청크를 4개씩 가져왔고, 1순위 RRF 점수({min(ref_top1):.2f}~{max(ref_top1):.2f})가 답변 가능 문항({min(ans_top1):.2f}~{max(ans_top1):.2f})과 겹칩니다. RRF 점수는 순위 기반으로 정규화된 값이라 "점수가 낮으면 검색 단계에서 미리 거절"하는 방식은 지금 상태로는 쓸 수 없습니다.</li>
</ul>
<b>보완 방향.</b> ① 검색 점수 대신 벡터 코사인 유사도 절대값(dense_similarity)에 임계값을 두어, 임계값 미만이면 LLM 호출 전에 거절(비용·지연 절감). ② 거절 문항을 15~20개로 늘리고 "관련은 있지만 문서에 없는 질문", "일부만 문서에 있는 질문", 오도성 전제 변형을 추가. ③ 이 거절 문항들을 회귀 테스트에 포함해 프롬프트·모델을 바꿔도 거절률이 유지되는지 확인.
<p class="note">한계: 5문항은 표본이 매우 작아 "5/5"가 거절률 100%를 보장하지 않습니다. 판사가 답변 모델과 같은 계열이라는 점도 앞 장과 같습니다.</p></div>
{ref_cards}"""

weak = sorted(rows, key=lambda r: r["context_precision"])[:3]
weak_html = "".join(
    f'<li><a href="#q{r["id"]}">Q{r["id"]:02d}</a> (CP {r["context_precision"]:.2f}, 관련 청크 순위 '
    f'{[i+1 for i, x in enumerate(r["relevance"]) if x]}) — {esc(r["question"][:50])}…</li>' for r in weak)

doc = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAG 성능 평가 보고서</title>
<style>
:root{{--bg:#f6f7f9;--fg:#1c2530;--mut:#667085;--card:#fff;--line:#e3e7ee;--good:#1a7f4b;--warn:#b7791f;--bad:#c0392b;--acc:#2f5fd0}}
@media (prefers-color-scheme:dark){{:root{{--bg:#12161c;--fg:#e8ecf2;--mut:#9aa6b6;--card:#1b2129;--line:#2b3440;--good:#4cc38a;--warn:#e0a84a;--bad:#ef7b6e;--acc:#7aa2ff}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 'Pretendard','Malgun Gothic',system-ui,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:28px 16px 80px}}h1{{font-size:26px;margin:0 0 4px}}h2{{font-size:19px;margin:36px 0 12px}}h3{{font-size:15.5px;margin:0}}h4{{font-size:13px;margin:14px 0 4px;color:var(--mut)}}
.sub{{color:var(--mut);margin:0 0 20px}}.grid{{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(190px,1fr))}}
.kpi,.box{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}}.kpi b{{display:block;font-size:30px;line-height:1.2}}.kpi span{{color:var(--mut);font-size:13px}}
.good{{color:var(--good)}}.warn{{color:var(--warn)}}.bad{{color:var(--bad)}}
table{{width:100%;border-collapse:collapse;font-size:13.5px}}th,td{{padding:7px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}}
.tbl{{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow-x:auto}}.tbl tr[onclick]{{cursor:pointer}}.tbl tr[onclick]:hover{{background:var(--bg)}}td.q{{min-width:240px}}th{{white-space:nowrap}}
.bar{{position:relative;height:20px;background:var(--bg);border-radius:6px;overflow:hidden;min-width:140px}}.bar i{{position:absolute;inset:0 auto 0 0;opacity:.35}}.bar i.good{{background:var(--good)}}.bar i.warn{{background:var(--warn)}}.bar i.bad{{background:var(--bad)}}.bar b{{position:relative;padding-left:8px;font-size:12.5px;line-height:20px}}
.dot{{display:inline-block;width:11px;height:11px;border-radius:50%;border:1.5px solid var(--mut);margin-right:3px}}.dot.on{{background:var(--good);border-color:var(--good)}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;margin:18px 0}}.meta{{color:var(--mut);font-size:12.5px;margin:4px 0 8px}}
.pill{{display:inline-block;border:1px solid currentColor;border-radius:99px;padding:1px 10px;font-size:12.5px;margin-right:6px;font-weight:600}}
.card img{{width:100%;border:1px solid var(--line);border-radius:10px;margin:12px 0}}.cols{{display:grid;gap:18px;grid-template-columns:1fr 1fr}}@media(max-width:760px){{.cols{{grid-template-columns:1fr}}}}
.claims{{padding-left:0;list-style:none;margin:0;font-size:13.5px}}.claims li{{padding:3px 0}}.claims .ok{{color:var(--fg)}}.claims .ng{{color:var(--bad);font-weight:600}}
.chunk{{font-size:13px;margin:4px 0}}.tag{{font-size:11px;border:1px solid var(--line);border-radius:4px;padding:0 6px;margin-right:6px;color:var(--mut)}}.tag.on{{color:var(--good);border-color:var(--good)}}
.note{{font-size:12.5px;color:var(--mut)}}li{{margin:3px 0}}a{{color:var(--acc)}}small{{color:var(--mut);font-weight:400}}.def{{font-size:13.5px;color:var(--mut)}}
</style></head><body><main>
<h1>RAG 웹서비스 성능 평가 보고서</h1>
<p class="sub">대상: 『한강 다리 이야기』 han-gang1-100.pdf (99쪽, 186청크) · 답변 가능 {n}개 + 거절 평가 {n_ref}개 = 총 35문항(han_gang_35.json) · 하이브리드 RRF(벡터+BM25), top-k 4 · 답변 모델 gemini-3.8-flash · 실행일 2026-10-28<br>
모든 질문은 실제 웹 UI(127.0.0.1:8000)에 입력했고 화면을 <code>rag/images/q01~q35.png</code>로 캡처했습니다.</p>

<h2>1. 평가 요소 4가지 점검</h2>
<div class="tbl"><table>
<tr><th>#</th><th>평가 요소</th><th>판정</th><th>근거</th></tr>
<tr><td>①</td><td>1-5쪽 적재 후 1-20쪽 투입 시 DB 중복 저장 여부 확인</td><td class="good"><b>확인 후 수정 완료</b></td><td>수정 전에는 1-5쪽 청크 7개가 중복 저장됐으나(35행/고유 28), 페이지 단위 증분 인덱싱과 content_hash 검사를 적용한 뒤 같은 실험에서 중복 0(28행/고유 28)을 확인(6장).</td></tr>
<tr><td>②</td><td>han-gang1-100.pdf 임베딩 + QA 쌍 테스트</td><td class="good"><b>충족</b></td><td>99쪽 → 186청크로 인덱싱(gemini-embedding-2, 3072차원). han_gang_35.json의 답변 가능 30문항 + 거절 평가 5문항, 총 35문항을 실제 웹 UI에 모두 입력하고 화면을 캡처(images/q01~q35.png). 단, 질문·정답은 Gemini가 PDF를 읽고 생성했고 사람 검수 기록은 없음(9장).</td></tr>
<tr><td>③</td><td>적절한 성과지표 선정</td><td class="good"><b>충족</b></td><td>검색은 Context Precision@4, 생성은 Faithfulness(3장). 보조로 Hit@4, 1순위 적중률, 응답 시간, 그리고 거절 평가(Negative Rejection Rate, 10장).</td></tr>
<tr><td>④</td><td>성능 수준 파악 + 개선·보완 방향</td><td class="good"><b>충족</b></td><td>CP 0.905, Faithfulness 0.995, 거절 5/5로 판정(2장, 10장)하고 개선 방향을 5장·10장에 정리.</td></tr>
</table></div>
<h2>2. 핵심 결과 (평가 요소 ②③)</h2>
<div class="grid">
<div class="kpi"><span>Context Precision@4 (평균)</span><b class="{tone(avg_cp)}">{avg_cp:.3f}</b><span>{grade(avg_cp)} · 검색 단계</span></div>
<div class="kpi"><span>Faithfulness (평균)</span><b class="{tone(avg_ff)}">{avg_ff:.3f}</b><span>{grade(avg_ff)} · 생성 단계</span></div>
<div class="kpi"><span>Hit@4</span><b class="good">{hit}/{n}</b><span>정답 근거 청크가 top-4에 포함</span></div>
<div class="kpi"><span>1순위 청크가 유용</span><b class="{tone(top1/n)}">{top1}/{n}</b><span>{top1/n*100:.0f}%</span></div>
<div class="kpi"><span>평균 응답 시간</span><b>{avg_lat:.1f}s</b><span>B4 생성 기준</span></div>
<div class="kpi"><span>근거 없는 주장</span><b class="{'good' if bad_claims<=3 else 'warn'}">{bad_claims}/{total_claims}</b><span>전체 claim 중</span></div>
</div>
<div class="box" style="margin-top:12px"><b>해석.</b> 검색은 30문항 모두 근거 청크를 top-4 안에 가져왔고(Hit@4 100%), 27문항은 1순위가 곧 정답 근거였습니다.
답변은 검색된 청크에 거의 전부 근거했습니다(주장 {total_claims}개 중 {bad_claims}개만 미지지). 즉 <b>현재 병목은 환각이 아니라 '정답 청크의 순위'</b>입니다.
관련 청크 개수는 순위별로 1위 {rank_rel[0]}/{n}, 2위 {rank_rel[1]}/{n}, 3위 {rank_rel[2]}/{n}, 4위 {rank_rel[3]}/{n}입니다.</div>

<h2>3. 사용한 지표 (조사 자료의 '우선 핵심 2개')</h2>
<div class="grid">
<div class="box"><b>① Context Precision@4 — 검색</b><p class="def">top-4 청크 각각이 참조 정답의 핵심 사실을 담고 있는지(유용=1/무관=0) 판정한 뒤, RAGAS 방식의 평균 정밀도(AP)로 계산합니다. 유용한 청크가 위쪽 순위에 있을수록 1.0에 가깝습니다. 예) 유용 위치 [1,0,1,0] → (1/1+2/3)/2 = 0.83</p></div>
<div class="box"><b>② Faithfulness — 생성</b><p class="def">생성 답변을 개별 주장(claim)으로 분해해, 각 주장이 검색된 청크만으로 뒷받침되는 비율입니다. 청크에 없는 내용을 지어내거나 모델 사전 지식을 섞으면 감점됩니다.</p></div>
</div>
<p class="def">판정은 LLM 판사(gemini-3.8-flash, temperature 0)가 수행했고, UI와 동일한 질의·top-k로 다시 검색한 청크 전문을 근거로 사용했습니다. 기준은 우수 ≥0.90, 양호 ≥0.70, 미흡 &lt;0.70 입니다.</p>

<h2>4. 그룹별 점수</h2>
<div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(440px,1fr))">
<div class="tbl"><table><tr><th>난이도</th><th>Context Precision</th><th>Faithfulness</th></tr>{bars(diff_items)}</table></div>
<div class="tbl"><table><tr><th>장</th><th>Context Precision</th><th>Faithfulness</th></tr>{bars(chap_items)}</table></div>
<div class="tbl"><table><tr><th>분류</th><th>Context Precision</th><th>Faithfulness</th></tr>{bars(cat_items)}</table></div>
</div>

<h2>5. 성능 판단과 개선·보완 방향 (평가 요소 ④)</h2>
<div class="box"><ul>{weak_html}</ul>
<p class="note">이 문항들은 정답 내용이 있는 청크가 3~4위로 밀려 있거나 같은 쪽의 인접 청크에 흩어져 있습니다.</p>
<p><b>성능 판단.</b> 현재 서비스는 이 문서·이 질문 유형에서 <b>실사용 가능한 수준</b>입니다(검색 {hit}/{n} 적중, 환각 거의 없음). 남은 약점은 ① 검색 순위, ② 중복 적재, ③ 평가 범위입니다.</p>
<ol>
<li><b>검색 순위 개선(Context Precision 0.905 → 0.95 목표):</b> top-k를 8~10으로 넓혀 가져온 뒤 리랭커(cross-encoder)로 재정렬해 상위 4개만 프롬프트에 사용. 복합 질문은 하위 질문으로 쪼개 각각 검색(쿼리 분해).</li>
<li><b>중복 적재 방지(적용 완료):</b> 6장의 페이지 단위 증분 인덱싱과 content_hash 검사로 해결. 다음 단계는 쪽 단위가 아닌 청크 단위 유사 중복(재스캔 문서) 처리.</li>
<li><b>평가 확대:</b> 거절 문항 확대(10장에서 5개 측정 → 15~20개로), Context Recall·Answer Correctness 병행, 다른 계열 모델로 교차 채점, 사람 검수로 정답 신뢰도 확보. 코드를 바꿀 때마다 이 30문항을 자동 재실행하는 회귀 테스트(DeepEval 등)도 권장.</li>
</ol></div>

<h2>6. 증분 인덱싱 vs 전체 재처리 (1-5쪽 → 1-20쪽) — 수정 전/후</h2>
<div class="box"><b>질문.</b> 1-5쪽이 이미 DB에 있을 때 1-20쪽이 들어오면, 전부 덮어쓰기보다 1-5쪽은 건너뛰고 6-20쪽만 임베딩하는 편이 낫지 않은가?
<p><b>결론: 제안하신 방향(이미 있는 쪽은 건너뛰기)이 맞고, 코드에 적용했습니다.</b> 수정 전에는 "덮어쓰기"도 "건너뛰기"도 아니고 1-5쪽을 중복으로 한 번 더 저장했습니다.</p></div>
<div class="tbl" style="margin-top:12px"><table>
<tr><th>실측 (임시 DB, 웹 UI와 같은 설정)</th><th>1-5쪽 투입</th><th>이어서 1-20쪽 투입 · 수정 전</th><th>이어서 1-20쪽 투입 · 수정 후</th><th>1-20쪽 한 번 더 (수정 후)</th></tr>
<tr><td>OCR 대상</td><td>5쪽 (24.7초)</td><td>20쪽 전체 (112.6초)</td><td class="good">새 15쪽(p.6~20)만 (94.8초)</td><td>건너뜀</td></tr>
<tr><td>임베딩 API 호출</td><td>7개</td><td>21개 (7개는 캐시 적중)</td><td>21개 (신규 쪽만)</td><td>0개</td></tr>
<tr><td>DB 청크 수 / 고유 청크 수</td><td>7 / 7</td><td class="bad"><b>35 / 28 (7개 중복)</b></td><td class="good"><b>28 / 28 (중복 0)</b></td><td>28 / 28</td></tr>
</table></div>
<div class="box" style="margin-top:12px"><ul>
<li><b>어떻게 막는가.</b> ① PDF 각 쪽의 원본(콘텐츠 스트림 + 삽입 이미지)을 SHA-256으로 지문화해 <code>indexed_pages</code> 테이블에 기록, ② 새 PDF가 들어오면 지문이 같은 쪽은 OCR 전에 제외하고 새 쪽만 부분 PDF로 만들어 OCR, ③ 모든 쪽이 이미 있으면 OCR·임베딩·저장을 전부 생략, ④ 청크 저장 시 <code>content_hash</code>가 이미 있으면 저장하지 않는 안전망. 같은 쪽이면 다른 PDF 파일에서도 지문이 일치함을 확인했습니다.</li>
<li><b>기존 DB도 대응.</b> 지문 기록이 없는 구버전 DB(현재 메인 DB)는 같은 폴더의 원본 PDF로 자동 보강합니다. 메인 DB 사본에서 1-20쪽·1-5쪽을 넣었을 때 OCR 없이 모두 건너뛰고 186→186청크를 유지했습니다.</li>
<li><b>시간 절감은 제한적입니다.</b> OCR이 쪽 수에 정확히 비례하지 않아 112.6초 → 94.8초(약 16%)였습니다. 이득의 중심은 <b>저장 중복 제거</b>와, 이미 있는 문서를 다시 넣을 때의 OCR·임베딩 전면 생략입니다.</li>
<li><b>"5쪽 끝과 6쪽 시작이 끊겨 품질이 떨어진다"는 우려는 이 시스템에 해당하지 않습니다.</b> 청크를 페이지별로 따로 나눠 만들기 때문에 쪽 경계를 넘는 청크가 원래 없습니다.</li>
<li><b>남은 한계.</b> 판별 단위가 "쪽 전체"라서, 같은 쪽이 편집·재스캔돼 바이트가 달라지면 새 쪽으로 처리됩니다. 이때는 content_hash 검사가 OCR 결과가 완전히 같은 청크만 걸러줍니다. 또 증분 추가된 쪽은 출처 파일명이 새 PDF 이름으로 기록됩니다.</li>
</ul></div>
<h2>7. 문항별 결과</h2>
<div class="tbl"><table><tr><th>#</th><th>장</th><th>분류</th><th>난이도</th><th>질문</th><th>유용 청크(1~4위)</th><th>CP</th><th>Faith</th></tr>{table}</table></div>

<h2>8. 문항별 화면 캡처와 판정 근거</h2>
{cards}

<h2>9. 한계</h2>
<div class="box"><ul>
<li>LLM 판사는 답변 모델과 같은 계열(Gemini)이라 관대하게 채점할 수 있습니다. Faithfulness {avg_ff:.3f}는 상한에 가까워 변별력이 낮으니, 다른 계열 모델로 교차 채점하면 더 믿을 수 있습니다.</li>
<li>답변 가능 문항 30개와 거절 문항 5개는 통계적으로 작은 표본이고, 모두 한 권의 책에서 나왔습니다. 거절 능력은 10장에서 5문항으로 처음 측정했으며, 표본이 작아 참고용입니다.</li>
<li>Context Precision은 참조 정답 기준의 LLM 판정이라, 판사가 '유용'의 범위를 넓게 잡으면 점수가 올라갑니다. 의심 문항은 6장의 청크 원문으로 직접 확인할 수 있습니다.</li>
</ul></div>
{refusal_section}
</main></body></html>"""

OUT.write_text(doc, encoding="utf-8")
print(f"saved {OUT} avg_cp={avg_cp:.3f} avg_ff={avg_ff:.3f} hit={hit} top1={top1} bad_claims={bad_claims}/{total_claims} lat={avg_lat:.1f}")
