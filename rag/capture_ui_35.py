"""실제 RAG 웹서비스(127.0.0.1:8000)에 han_gang_35.json 30문항을 질의하고 UI 스크린샷을 저장한다.

- 스크린샷: rag/images/q01.png ~ q30.png ([B단계] 질문에 답하기 섹션)
- 원본 응답: rag/ui_run_35.json (답변, 검색 청크 미리보기, 소요시간)
"""

import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

CURRENT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = CURRENT_DIR / "images"
OUT_JSON = CURRENT_DIR / "ui_run_35.json"
URL = "http://127.0.0.1:8000/"
TOP_K = 4

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    questions = json.loads((CURRENT_DIR / "han_gang_35.json").read_text(encoding="utf-8"))
    only = {int(a) for a in sys.argv[1:]}
    results = json.loads(OUT_JSON.read_text(encoding="utf-8")) if OUT_JSON.exists() else {}
    IMAGES_DIR.mkdir(exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        page.goto(URL, wait_until="networkidle")

        for q in questions:
            qid = q["id"]
            if only and qid not in only:
                continue
            if str(qid) in results and not only:
                continue
            print(f"[Q{qid:02d}] {q['question'][:40]}...", flush=True)

            page.fill("#queryInput", q["question"])
            page.fill("#topKInput", str(TOP_K))
            t0 = time.time()
            with page.expect_response(lambda r: "/api/ask" in r.url, timeout=120000) as resp_info:
                page.click("#btnAsk")
            resp = resp_info.value
            data = resp.json()
            page.wait_for_function(
                "() => document.getElementById('answerLatency').innerText.length > 0 "
                "|| document.getElementById('finalAnswerBox').innerText.includes('오류')",
                timeout=30000,
            )
            time.sleep(0.5)

            section = page.locator("#askForm").locator("xpath=ancestor::section[1] | ancestor::div[contains(@class,'rounded')][1]").first
            png = IMAGES_DIR / f"q{qid:02d}.png"
            try:
                section.screenshot(path=str(png))
            except Exception:
                page.screenshot(path=str(png), full_page=True)

            results[str(qid)] = {
                "status": resp.status,
                "wall_time": round(time.time() - t0, 2),
                "response": data,
            }
            OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            ans = data.get("answer", data.get("detail", ""))
            print(f"   status={resp.status} {str(ans)[:70]!r}", flush=True)
            time.sleep(1.0)

        browser.close()


if __name__ == "__main__":
    main()
