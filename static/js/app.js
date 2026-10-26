let currentThreadId = "";
let messageCount = 0;
let currentModel = "gemini";
let currentReasoningEffort = "medium";
let currentModelLabel = "Flash";

// ==============================================================================
// [1] Run Settings (우측 패널 & 출력 토큰 수 제어)
// ==============================================================================

// 우측 사이드바 열기/닫기
function toggleRightPanel() {
  const panel = document.getElementById("rightPanel");
  if (panel) {
    panel.classList.toggle("collapsed");
  }
}

// 섹션 접기/펼치기
function toggleSection(contentId, caretId) {
  const content = document.getElementById(contentId);
  const caret = document.getElementById(caretId);
  if (!content) return;
  if (content.style.display === "none") {
    content.style.display = "flex";
    if (caret) caret.textContent = "▾";
  } else {
    content.style.display = "none";
    if (caret) caret.textContent = "▸";
  }
}

// ★ Output length (출력 토큰 제한) 양방향 동기화
function syncOutputLength(source) {
  const slider = document.getElementById("outputLengthSlider");
  const input = document.getElementById("outputLengthInput");
  const badge = document.getElementById("tokenBadge");
  const headerBadge = document.getElementById("badgeMaxTokens");

  let val = 2048;
  if (source === "slider") {
    val = parseInt(slider.value, 10);
    input.value = val;
  } else if (source === "input") {
    val = parseInt(input.value, 10) || 64;
    val = Math.max(64, Math.min(8192, val));
    slider.value = val;
  } else if (source === "preset") {
    val = parseInt(input.value, 10);
  }

  const formatted = val.toLocaleString();
  if (badge) badge.textContent = `${formatted} 토큰`;
  if (headerBadge) headerBadge.textContent = `Max ${formatted}T`;

  // 프리셋 버튼 active 상태 업데이트
  document.querySelectorAll(".token-presets .preset-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.textContent.includes(String(val)));
  });

  localStorage.setItem("chat_output_length", val);
}

// 토큰 프리셋 버튼 클릭
function setTokenPreset(tokens) {
  const slider = document.getElementById("outputLengthSlider");
  const input = document.getElementById("outputLengthInput");
  if (slider) slider.value = tokens;
  if (input) input.value = tokens;
  syncOutputLength("preset");
}

// Temperature 양방향 동기화
function syncTemp(source) {
  const slider = document.getElementById("tempSlider");
  const input = document.getElementById("tempInput");
  if (source === "slider") {
    input.value = slider.value;
  } else {
    let v = parseFloat(input.value) || 0.0;
    v = Math.max(0.0, Math.min(2.0, v));
    slider.value = v;
  }
}

// Thinking level 변경 시 Temperature 및 프롬프트 캡슐 연동
function updateThinkingLevel() {
  const level = document.getElementById("thinkingLevel").value;
  const slider = document.getElementById("tempSlider");
  const input = document.getElementById("tempInput");
  let temp = 0.7;
  let label = "Flash";
  if (level === "low") {
    temp = 0.2;
    label = "빠른 추론";
  } else if (level === "medium") {
    temp = 0.7;
    label = "Flash";
  } else if (level === "high") {
    temp = 1.0;
    label = "사고 모델";
  }

  if (slider && input) {
    slider.value = temp;
    input.value = temp;
  }

  const model = document.querySelector('input[name="model"]:checked')?.value || "gemini";
  if (model === "gemini") {
    selectModelEffort(model, level, label);
  }
}

// ==============================================================================
// [1-1] 하단 프롬프트 모델 & 추론 모드(Reasoning Effort) 팝업 드롭다운
// ==============================================================================

// 드롭다운 메뉴 열기/닫기 토글
function toggleModelDropdown(event) {
  if (event) event.stopPropagation();
  const menu = document.getElementById("modelPopupMenu");
  const caret = document.getElementById("capsuleCaret");
  if (!menu) return;
  const isHidden = menu.style.display === "none" || menu.style.display === "";
  menu.style.display = isHidden ? "block" : "none";
  if (caret) caret.textContent = isHidden ? "▴" : "▾";
}

// 모델 및 추론 수준(Reasoning Effort) 선택
function selectModelEffort(model, effort, label) {
  currentModel = model;
  currentReasoningEffort = effort || "medium";
  currentModelLabel = label;

  // 1. 하단 캡슐 버튼 라벨 갱신
  const capsuleLabel = document.getElementById("modelCapsuleLabel");
  if (capsuleLabel) capsuleLabel.textContent = label;

  // 2. 팝업 메뉴 내 체크마크(✓) 및 활성화 상태 갱신
  document.querySelectorAll(".popup-option-item").forEach((el) => {
    el.classList.remove("active");
  });
  if (model === "ollama") {
    document.getElementById("popOpt_ollama")?.classList.add("active");
  } else if (effort === "high") {
    document.getElementById("popOpt_thinking")?.classList.add("active");
  } else if (effort === "low") {
    document.getElementById("popOpt_fast")?.classList.add("active");
  } else {
    document.getElementById("popOpt_flash")?.classList.add("active");
  }

  // 3. 우측 Run settings 패널의 라디오 및 셀렉트 동기화
  const radio = document.querySelector(`input[name="model"][value="${model}"]`);
  if (radio) radio.checked = true;

  const selThinking = document.getElementById("thinkingLevel");
  if (selThinking && effort && model === "gemini") {
    selThinking.value = effort;
  }

  // 4. 팝업 메뉴 닫기
  const menu = document.getElementById("modelPopupMenu");
  if (menu) menu.style.display = "none";
  const caret = document.getElementById("capsuleCaret");
  if (caret) caret.textContent = "▾";

  // 5. 로컬 스토리지에 사용자 선택 저장
  localStorage.setItem("chat_model", model);
  localStorage.setItem("chat_reasoning_effort", currentReasoningEffort);
  localStorage.setItem("chat_model_label", label);

  // 6. 상단 뱃지 동기화
  updateModelUI();
}

// 도구(구글 검색) 토글 제어
function onSearchToggle(mode) {
  const grounding = document.getElementById("toggleGrounding");
  const cse = document.getElementById("toggleCSE");

  if (mode === "grounding" && grounding.checked) {
    if (cse) cse.checked = false;
  } else if (mode === "cse" && cse.checked) {
    if (grounding) grounding.checked = false;
  }
  updateModelUI();
}

function getSelectedSearchMode() {
  const grounding = document.getElementById("toggleGrounding");
  const cse = document.getElementById("toggleCSE");
  if (grounding && grounding.checked) return "grounding";
  if (cse && cse.checked) return "cse";
  return "off";
}

// ==============================================================================
// [1-2] 구조화된 출력 (Structured Outputs - One-Shot 예시 포맷팅)
// ==============================================================================

const STRUCTURED_PRESETS = {
  news: {
    input: "링크 페이지 : https://www.hani.co.kr/arti/international/international_general/1284117.html",
    output: "뉴스 제목 : \n기자 이름 : \n뉴스 내용 요약 : "
  },
  summary_text: {
    input: "인공지능 트렌드 알려줘",
    output: "■ 주제 : \n■ 핵심 1줄 요약 : \n■ 주요 포인트 :\n1. \n2. \n3. \n■ 시사점 및 결론 : "
  },
  qa: {
    input: "파이썬에서 가상환경을 만드는 방법은?",
    output: "Q. 질문 요지 : \nA. 핵심 답변 : \n- 실행 명령어 : \n- 주의사항 : "
  },
  json_summary: {
    input: "인공지능 트렌드 알려줘",
    output: JSON.stringify({
      topic: "주제",
      summary: "핵심 1줄 요약",
      key_points: [
        "핵심 포인트 1",
        "핵심 포인트 2",
        "핵심 포인트 3"
      ],
      conclusion: "결론 또는 시사점"
    }, null, 2)
  },
  custom: {
    input: "",
    output: ""
  }
};

// 설정 카드 열기/닫기 토글
function toggleStructuredConfig() {
  const area = document.getElementById("structuredConfigArea");
  const btn = document.getElementById("btnToggleStructuredConfig");
  if (!area) return;
  const isHidden = area.style.display === "none";
  area.style.display = isHidden ? "flex" : "none";
  if (btn) btn.textContent = isHidden ? "설정 ▴" : "설정 ▾";
}

// 스위치 토글 이벤트
function onStructuredToggle() {
  const toggle = document.getElementById("toggleStructuredOutput");
  const area = document.getElementById("structuredConfigArea");
  const btn = document.getElementById("btnToggleStructuredConfig");
  const badge = document.getElementById("badgeStructured");

  if (toggle && toggle.checked) {
    if (area) area.style.display = "flex";
    if (btn) btn.textContent = "설정 ▴";
    if (badge) badge.style.display = "inline-block";
  } else {
    if (area) area.style.display = "none";
    if (btn) btn.textContent = "설정 ▾";
    if (badge) badge.style.display = "none";
  }
}

// 프리셋 선택 시 예시 입력/출력 갱신
function applyStructuredPreset() {
  const sel = document.getElementById("presetSelect");
  if (!sel) return;
  const presetKey = sel.value;
  const preset = STRUCTURED_PRESETS[presetKey];
  if (!preset) return;

  const inputEl = document.getElementById("oneShotInput");
  const outputEl = document.getElementById("oneShotOutput");
  if (inputEl && preset.input !== undefined) inputEl.value = preset.input;
  if (outputEl && preset.output !== undefined) outputEl.value = preset.output;
}

// 템플릿 복원 함수
function resetOneShotFormat() {
  applyStructuredPreset();
}

// JSON 정렬 버튼 (JSON 작성 시 편의용)
function formatOneShotJson() {
  const outputEl = document.getElementById("oneShotOutput");
  if (!outputEl) return;
  try {
    const parsed = JSON.parse(outputEl.value);
    outputEl.value = JSON.stringify(parsed, null, 2);
  } catch (e) {
    alert("유효한 JSON 형식이 아닙니다: " + e.message);
  }
}

// 일반 텍스트 클립보드 복사
async function copyCardText(cardId, btn) {
  const text = window._cachedCardTexts && window._cachedCardTexts[cardId];
  if (text) {
    try {
      await navigator.clipboard.writeText(text);
      const orig = btn.textContent;
      btn.textContent = "✅ 복사 완료!";
      setTimeout(() => { btn.textContent = orig; }, 1500);
    } catch (e) {
      alert("복사 실패: " + e.message);
    }
  }
}

// 클립보드 복사
async function copyJsonToClipboard(btn, jsonText) {
  try {
    await navigator.clipboard.writeText(jsonText);
    const orig = btn.textContent;
    btn.textContent = "✅ 복사 완료!";
    setTimeout(() => { btn.textContent = orig; }, 1500);
  } catch (e) {
    alert("복사 실패: " + e.message);
  }
}

// 캐시된 JSON 복사 헬퍼
function copyCachedJson(cardId, btn) {
  const jsonText = window._cachedJsonReplies && window._cachedJsonReplies[cardId];
  if (jsonText) {
    copyJsonToClipboard(btn, jsonText);
  }
}

// 봇 답변 파싱 및 카드(일반 텍스트 또는 JSON) 렌더링
function renderBotReplyContent(replyText, isStructured) {
  if (!replyText) return "";

  // 1. JSON 코드 블록 또는 원시 JSON 파싱 검사
  let cleanJson = null;
  const jsonBlockRegex = /```(?:json)?\s*([\s\S]*?)\s*```/;
  const match = replyText.match(jsonBlockRegex);

  if (match) {
    try {
      const parsed = JSON.parse(match[1]);
      cleanJson = JSON.stringify(parsed, null, 2);
    } catch (e) {}
  } else if (replyText.trim().startsWith("{") && replyText.trim().endsWith("}")) {
    try {
      const parsed = JSON.parse(replyText.trim());
      cleanJson = JSON.stringify(parsed, null, 2);
    } catch (e) {}
  }

  // 1-1. JSON 형식일 경우: 다크 코드 카드 렌더링
  if (cleanJson) {
    const cardId = "jsonCard_" + Date.now() + "_" + Math.floor(Math.random() * 1000);
    window._cachedJsonReplies = window._cachedJsonReplies || {};
    window._cachedJsonReplies[cardId] = cleanJson;

    return `
      <div class="json-result-card" id="${cardId}">
        <div class="json-header-bar">
          <span>📋 <b>Structured Output (JSON)</b></span>
          <button type="button" class="btn-copy-json" onclick="copyCachedJson('${cardId}', this)">📋 JSON 복사</button>
        </div>
        <pre style="margin:0; overflow-x:auto;"><code>${escapeHtml(cleanJson)}</code></pre>
      </div>
    `;
  }

  // 1-2. 구조화된 출력(isStructured=true)일 경우: 전용 서식 카드 렌더링
  if (isStructured) {
    const cardId = "structCard_" + Date.now() + "_" + Math.floor(Math.random() * 1000);
    window._cachedCardTexts = window._cachedCardTexts || {};
    window._cachedCardTexts[cardId] = replyText;

    return `
      <div class="structured-result-card" id="${cardId}">
        <div class="structured-card-header">
          <span>📋 <b>구조화된 텍스트 출력</b></span>
          <button type="button" class="btn-copy-card" onclick="copyCardText('${cardId}', this)">📋 텍스트 복사</button>
        </div>
        <div class="structured-card-body">${escapeHtml(replyText)}</div>
      </div>
    `;
  }

  return escapeHtml(replyText);
}

// ==============================================================================
// [1-3] Code Execution (파이썬 코드 실행) 툴 헬퍼
// ==============================================================================

function onCodeExecutionToggle() {
  const toggle = document.getElementById("toggleCodeExecution");
  const badge = document.getElementById("badgeCodeExecution");
  if (badge) {
    badge.style.display = toggle && toggle.checked ? "inline-block" : "none";
  }
}

function toggleCodeExecCard(headerEl) {
  const card = headerEl.closest(".code-execution-card");
  const body = card.querySelector(".code-exec-body");
  const caret = card.querySelector(".code-exec-caret");
  if (!body) return;
  if (body.style.display === "none") {
    body.style.display = "flex";
    if (caret) caret.textContent = "▾";
  } else {
    body.style.display = "none";
    if (caret) caret.textContent = "▸";
  }
}

async function copyCodeSnippet(btn) {
  const item = btn.closest(".code-exec-item");
  const codeEl = item ? item.querySelector(".code-snippet code") : null;
  if (codeEl) {
    try {
      await navigator.clipboard.writeText(codeEl.textContent);
      const orig = btn.textContent;
      btn.textContent = "✅ 복사됨";
      setTimeout(() => { btn.textContent = orig; }, 1500);
    } catch (e) {
      alert("코드 복사 실패: " + e.message);
    }
  }
}

function renderCodeExecutionsHtml(executions) {
  if (!executions || executions.length === 0) return "";

  let itemsHtml = "";
  executions.forEach((ex, idx) => {
    const labelNum = executions.length > 1 ? ` #${idx + 1}` : "";
    itemsHtml += `
      <div class="code-exec-item">
        <div class="code-exec-label">
          <span>📝 파이썬 실행 코드${labelNum}:</span>
          <button type="button" class="btn-copy-code" onclick="copyCodeSnippet(this)">코드 복사</button>
        </div>
        <pre class="code-snippet"><code>${escapeHtml(ex.code)}</code></pre>
        <div class="code-exec-label" style="margin-top:6px;">
          <span>⚙️ 실행 결과 (Stdout / Output):</span>
        </div>
        <pre class="output-snippet"><code>${escapeHtml(ex.output)}</code></pre>
      </div>
    `;
  });

  return `
    <div class="code-execution-card">
      <div class="code-exec-header" onclick="toggleCodeExecCard(this)" title="코드 및 실행 결과 펼치기/접기">
        <div style="display:flex; align-items:center; gap:6px;">
          <span style="font-size:0.9rem;">⚡</span>
          <span class="code-exec-title">Code Execution (Python 코드 실행 ${executions.length}건)</span>
        </div>
        <span class="code-exec-caret">▾</span>
      </div>
      <div class="code-exec-body">
        ${itemsHtml}
      </div>
    </div>
  `;
}

// ==============================================================================
// [2] 대화 세션 및 메시지 처리
// ==============================================================================

// 새 대화 세션 시작
async function resetChat() {
  try {
    const res = await fetch("/api/reset", { method: "POST" });
    const data = await res.json();
    currentThreadId = data.thread_id;
    localStorage.setItem("chat_thread_id", currentThreadId);
    document.getElementById("infoThread").textContent = currentThreadId;
    const linkH = document.getElementById("linkHistory");
    if (linkH) linkH.href = "/api/history/" + encodeURIComponent(currentThreadId);
    messageCount = 0;
    document.getElementById("infoMsgCount").textContent = "0개 메시지";

    // 화면 초기화
    const chatBox = document.getElementById("chatBox");
    chatBox.innerHTML = `
      <div class="msg-row bot">
        <div class="avatar bot">🤖</div>
        <div class="bubble">새 대화 세션이 시작되었습니다. 무엇이든 물어보세요!</div>
      </div>
    `;
    document.getElementById("metricLastInput").textContent = "0 토큰";
    document.getElementById("metricLastOutput").textContent = "0 토큰";
    document.getElementById("metricTotalTokens").textContent = "0 토큰";
    document.getElementById("metricTotalCost").textContent = "0.000원";

    loadSessionsList();
  } catch (e) {
    alert("세션 초기화 실패: " + e);
  }
}

// 시간 포맷 (MM/DD HH:mm)
function formatTime(isoStr) {
  if (!isoStr) return "";
  try {
    const d = new Date(isoStr);
    if (isNaN(d.getTime())) return "";
    const m = (d.getMonth() + 1).toString().padStart(2, "0");
    const day = d.getDate().toString().padStart(2, "0");
    const h = d.getHours().toString().padStart(2, "0");
    const min = d.getMinutes().toString().padStart(2, "0");
    return `${m}/${day} ${h}:${min}`;
  } catch (e) {
    return "";
  }
}

// 대화 세션 목록 불러오기 (좌측 사이드바)
async function loadSessionsList() {
  const container = document.getElementById("sessionsList");
  if (!container) return;
  try {
    const res = await fetch("/api/sessions");
    if (!res.ok) return;
    const data = await res.json();
    const sessions = data.sessions || [];

    if (sessions.length === 0) {
      container.innerHTML = '<div class="sessions-empty">저장된 대화가 없습니다.</div>';
      return;
    }

    container.innerHTML = "";
    sessions.forEach((s) => {
      const item = document.createElement("div");
      const isActive = s.thread_id === currentThreadId;
      item.className = "session-item" + (isActive ? " active" : "");
      item.onclick = () => switchSession(s.thread_id);

      const rawTitle = s.title || `대화 (${s.thread_id})`;
      const title = escapeHtml(rawTitle);
      const safeTitleAttr = escapeAttr(rawTitle);
      const timeStr = formatTime(s.updated_at);
      const totalTok = (s.total_input || 0) + (s.total_output || 0);
      const tokStr = totalTok > 0 ? `${totalTok.toLocaleString()}T` : "";
      const costStr = s.total_cost_krw > 0 ? `${s.total_cost_krw.toFixed(2)}원` : "";
      const metaParts = [timeStr, tokStr, costStr].filter(Boolean);
      const metaStr = metaParts.length > 0 ? metaParts.join(" · ") : "대화 진행 전";

      item.innerHTML = `
        <div class="session-item-content">
          <div class="session-item-title" title="${title} (더블클릭하여 이름 변경)" ondblclick="renameSession(event, '${escapeHtml(s.thread_id)}', '${safeTitleAttr}')">💬 ${title}</div>
          <div class="session-item-meta">
            <span><code>${escapeHtml(s.thread_id)}</code></span>
            <span>${escapeHtml(metaStr)}</span>
          </div>
        </div>
        <div class="session-item-actions">
          <button class="btn-action-session btn-edit-session" title="이름 변경" onclick="renameSession(event, '${escapeHtml(s.thread_id)}', '${safeTitleAttr}')">✏️</button>
          <button class="btn-action-session btn-del-session" title="대화 삭제" onclick="deleteSession(event, '${escapeHtml(s.thread_id)}')">🗑️</button>
        </div>
      `;
      container.appendChild(item);
    });
  } catch (e) {
    console.error("세션 목록 로드 실패:", e);
  }
}

// 대화 세션 이름 변경
async function renameSession(event, threadId, currentTitle) {
  if (event) event.stopPropagation();
  const rawTitle = currentTitle || "새 대화";
  const newTitle = prompt("대화 세션의 새로운 이름을 입력하세요:", rawTitle);
  if (newTitle === null) return;
  const trimmed = newTitle.trim();
  if (!trimmed) {
    alert("대화 제목을 1자 이상 입력해 주세요.");
    return;
  }
  if (trimmed === rawTitle) return;

  try {
    const res = await fetch("/api/sessions/" + encodeURIComponent(threadId), {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: trimmed }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "이름 변경 실패");
    }
    await loadSessionsList();
  } catch (e) {
    alert("대화 이름 변경 중 오류: " + e.message);
  }
}

function escapeAttr(str) {
  if (!str) return "";
  return str.replace(/\\/g, "\\\\").replace(/'/g, "\\'").replace(/"/g, "&quot;");
}

// 대화 세션 전환
async function switchSession(threadId) {
  if (threadId === currentThreadId) return;
  try {
    const res = await fetch("/api/session/" + encodeURIComponent(threadId));
    if (!res.ok) throw new Error("세션 불러오기 실패");
    const data = await res.json();

    currentThreadId = data.thread_id;
    localStorage.setItem("chat_thread_id", currentThreadId);
    document.getElementById("infoThread").textContent = currentThreadId;
    const linkH = document.getElementById("linkHistory");
    if (linkH) linkH.href = "/api/history/" + encodeURIComponent(currentThreadId);

    const chatBox = document.getElementById("chatBox");
    chatBox.innerHTML = "";

    if (data.messages && data.messages.length > 0) {
      data.messages.forEach((m) => {
        const isUser = m.type === "human";
        const row = document.createElement("div");
        row.className = "msg-row " + (isUser ? "user" : "bot");
        row.innerHTML = `
          <div class="avatar ${isUser ? "user" : "bot"}">${isUser ? "👤" : "🤖"}</div>
          <div class="bubble">${escapeHtml(m.content)}</div>
        `;
        chatBox.appendChild(row);
      });
      messageCount = data.messages.length;
    } else {
      chatBox.innerHTML = `
        <div class="msg-row bot">
          <div class="avatar bot">🤖</div>
          <div class="bubble">이전 메시지가 없는 세션입니다. 질문을 입력해보세요!</div>
        </div>
      `;
      messageCount = 0;
    }
    chatBox.scrollTop = chatBox.scrollHeight;
    document.getElementById("infoMsgCount").textContent = messageCount + "개 메시지";

    const st = data.token_stats || { total_tokens: 0, total_cost_krw: 0, total_cost_usd: 0 };
    document.getElementById("metricLastInput").textContent = "-";
    document.getElementById("metricLastOutput").textContent = "-";
    document.getElementById("metricTotalTokens").textContent = (st.total_tokens || 0).toLocaleString() + " 토큰";
    document.getElementById("metricTotalCost").textContent = (st.total_cost_krw || 0).toFixed(3) + "원 ($" + (st.total_cost_usd || 0).toFixed(5) + ")";

    loadSessionsList();
  } catch (e) {
    alert("대화를 불러오지 못했습니다: " + e.message);
  }
}

// 대화 세션 삭제
async function deleteSession(event, threadId) {
  event.stopPropagation();
  if (!confirm(`'${threadId}' 대화 세션을 삭제하시겠습니까?`)) return;

  try {
    const res = await fetch("/api/sessions/" + encodeURIComponent(threadId), { method: "DELETE" });
    if (!res.ok) throw new Error("삭제 실패");

    if (threadId === currentThreadId) {
      await resetChat();
    } else {
      await loadSessionsList();
    }
  } catch (e) {
    alert("삭제 중 오류 발생: " + e.message);
  }
}

// 모델 / 검색 UI 상태 업데이트
function updateModelUI() {
  const model = document.querySelector('input[name="model"]:checked')?.value || currentModel || "gemini";
  currentModel = model;
  const searchMode = getSelectedSearchMode();

  const badgeModel = document.getElementById("badgeModel");
  const badgeSearch = document.getElementById("badgeSearch");
  const badgeStruct = document.getElementById("badgeStructured");
  const toggleStruct = document.getElementById("toggleStructuredOutput");

  // 헤더 모델 및 추론 수준 뱃지 갱신
  if (badgeModel) {
    if (model === "ollama") {
      badgeModel.textContent = "Ollama (gemma4:e2b)";
    } else {
      if (currentReasoningEffort === "high") {
        badgeModel.textContent = "Gemini 2.5 사고 모델 (High)";
      } else if (currentReasoningEffort === "low") {
        badgeModel.textContent = "Gemini 2.5 Flash (Low)";
      } else {
        badgeModel.textContent = "Gemini 2.5 Flash";
      }
    }
  }

  // 하단 캡슐 버튼 라벨 갱신
  const capsuleLabel = document.getElementById("modelCapsuleLabel");
  if (capsuleLabel) {
    if (model === "ollama") {
      capsuleLabel.textContent = "Ollama";
    } else if (currentReasoningEffort === "high") {
      capsuleLabel.textContent = "사고 모델";
    } else if (currentReasoningEffort === "low") {
      capsuleLabel.textContent = "빠른 추론";
    } else {
      capsuleLabel.textContent = "Flash";
    }
  }

  // 팝업 옵션 활성화 상태 동기화
  document.querySelectorAll(".popup-option-item").forEach((el) => {
    el.classList.remove("active");
  });
  if (model === "ollama") {
    document.getElementById("popOpt_ollama")?.classList.add("active");
  } else if (currentReasoningEffort === "high") {
    document.getElementById("popOpt_thinking")?.classList.add("active");
  } else if (currentReasoningEffort === "low") {
    document.getElementById("popOpt_fast")?.classList.add("active");
  } else {
    document.getElementById("popOpt_flash")?.classList.add("active");
  }

  if (badgeSearch) {
    if (searchMode === "grounding") {
      badgeSearch.textContent = "🌐 Grounding";
      badgeSearch.className = "badge search-on";
    } else if (searchMode === "cse") {
      badgeSearch.textContent = "🔎 Custom Search";
      badgeSearch.className = "badge search-on";
    } else {
      badgeSearch.textContent = "검색 OFF";
      badgeSearch.className = "badge";
    }
  }

  if (badgeStruct && toggleStruct) {
    badgeStruct.style.display = toggleStruct.checked ? "inline-block" : "none";
  }

  const badgeCode = document.getElementById("badgeCodeExecution");
  const toggleCode = document.getElementById("toggleCodeExecution");
  if (badgeCode && toggleCode) {
    badgeCode.style.display = toggleCode.checked ? "inline-block" : "none";
  }
}

// 메시지 전송 (Output length 및 Temperature, Structured Output, Code Execution 포함)
async function sendMessage() {
  const input = document.getElementById("userInput");
  const msg = input.value.trim();
  if (!msg) return;

  const model = document.querySelector('input[name="model"]:checked').value;
  const searchMode = getSelectedSearchMode();
  const maxTokens = parseInt(document.getElementById("outputLengthInput")?.value || "2048", 10);
  const temp = parseFloat(document.getElementById("tempInput")?.value || "0.7");

  // 구조화된 출력 (Structured outputs) 설정 확인
  const toggleStructured = document.getElementById("toggleStructuredOutput");
  const structuredOutput = toggleStructured ? toggleStructured.checked : false;
  const oneShotInput = document.getElementById("oneShotInput")?.value || "";
  const oneShotOutput = document.getElementById("oneShotOutput")?.value || "";

  // 코드 실행 (Code execution) 설정 확인
  const toggleCode = document.getElementById("toggleCodeExecution");
  const codeExecution = toggleCode ? toggleCode.checked : false;

  const btn = document.getElementById("btnSend");
  const chatBox = document.getElementById("chatBox");

  // 사용자 메시지 렌더링
  input.value = "";
  btn.disabled = true;

  const userRow = document.createElement("div");
  userRow.className = "msg-row user";
  userRow.innerHTML = `
    <div class="avatar user">👤</div>
    <div class="bubble">${escapeHtml(msg)}</div>
  `;
  chatBox.appendChild(userRow);

  // 로딩 말풍선 표시
  const loadingRow = document.createElement("div");
  loadingRow.className = "msg-row bot";
  loadingRow.id = "loadingMsg";
  loadingRow.innerHTML = `
    <div class="avatar bot">🤖</div>
    <div class="bubble">
      <div class="typing"><span></span><span></span><span></span></div>
    </div>
  `;
  chatBox.appendChild(loadingRow);
  chatBox.scrollTop = chatBox.scrollHeight;

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: msg,
        model: model,
        search_mode: searchMode,
        thread_id: currentThreadId || null,
        max_output_tokens: maxTokens,
        temperature: temp,
        reasoning_effort: currentReasoningEffort,
        structured_output: structuredOutput,
        one_shot_input: oneShotInput,
        one_shot_output: oneShotOutput,
        code_execution: codeExecution,
      }),
    });

    if (!res.ok) {
      let errMsg = "서버 응답 오류";
      try {
        const errData = await res.json();
        errMsg = errData.detail || JSON.stringify(errData);
      } catch (e) {
        const txt = await res.text();
        errMsg = txt || `서버 오류 (${res.status}: ${res.statusText})`;
      }
      throw new Error(errMsg);
    }

    const data = await res.json();
    currentThreadId = data.thread_id;
    localStorage.setItem("chat_thread_id", currentThreadId);
    document.getElementById("infoThread").textContent = currentThreadId;
    const linkH = document.getElementById("linkHistory");
    if (linkH) linkH.href = "/api/history/" + encodeURIComponent(currentThreadId);

    // 로딩 제거
    loadingRow.remove();

    // 출처 링크 및 캐시 적중 여부 표시
    let sourcesHtml = "";
    if (data.sources && data.sources.length > 0) {
      const isHit = Boolean(data.cache_hit) || data.sources.some(s => s.cached);
      const boxClass = isHit ? "sources-box cache-hit-box" : "sources-box";
      const headerText = isHit
        ? "⚡ <b>LangGraph Store 캐시 HIT</b> (동일 질의 재사용 · API 호출 절약):"
        : "🌐 <b>실시간 검색 출처</b> (LangGraph Store에 30분간 캐싱):";
      sourcesHtml = `<div class="${boxClass}">${headerText}<br>`;
      data.sources.forEach((s) => {
        const badge = s.cached ? " <span style='font-size:0.7rem; color:#16a34a; font-weight:700;'>[⚡캐시]</span>" : "";
        sourcesHtml += `<a class="source-link" href="${s.uri}" target="_blank" rel="noopener">🔗 ${escapeHtml(s.title)}${badge}</a>`;
      });
      sourcesHtml += '</div>';
    }

    const codeExecHtml = renderCodeExecutionsHtml(data.code_executions);
    const botReplyHtml = renderBotReplyContent(data.reply, Boolean(data.is_structured));
    const botRow = document.createElement("div");
    botRow.className = "msg-row bot";
    botRow.innerHTML = `
      <div class="avatar bot">🤖</div>
      <div class="bubble">
        ${codeExecHtml}
        ${botReplyHtml}
        ${sourcesHtml}
      </div>
    `;
    chatBox.appendChild(botRow);
    chatBox.scrollTop = chatBox.scrollHeight;

    // 메트릭 업데이트
    const stats = data.token_stats;
    document.getElementById("metricLastInput").textContent = stats.last_input.toLocaleString() + " 토큰";
    document.getElementById("metricLastOutput").textContent = stats.last_output.toLocaleString() + " 토큰";
    document.getElementById("metricTotalTokens").textContent = stats.total_tokens.toLocaleString() + " 토큰";
    if (model === "ollama") {
      document.getElementById("metricTotalCost").textContent = "0원 (로컬 무료)";
    } else {
      document.getElementById("metricTotalCost").textContent = stats.total_cost_krw.toFixed(3) + "원 ($" + stats.total_cost_usd.toFixed(5) + ")";
    }

    // 메시지 카운트
    messageCount += 2;
    document.getElementById("infoMsgCount").textContent = messageCount + "개 메시지";

    // 사이드바 세션 목록 및 캐시 통계 갱신
    loadSessionsList();
    loadCacheStats();

  } catch (err) {
    loadingRow.remove();
    const errRow = document.createElement("div");
    errRow.className = "msg-row bot";
    errRow.innerHTML = `
      <div class="avatar bot">⚠️</div>
      <div class="bubble" style="background:#fee2e2; color:#b91c1c;">
        오류 발생: ${escapeHtml(err.message)}
      </div>
    `;
    chatBox.appendChild(errRow);
  } finally {
    btn.disabled = false;
    input.focus();
  }
}

// HTML 이스케이프
function escapeHtml(str) {
  if (!str) return "";
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// 초기 세션 복원
async function restoreSession() {
  const saved = localStorage.getItem("chat_thread_id");
  if (!saved) { return resetChat(); }
  try {
    const res = await fetch("/api/session/" + encodeURIComponent(saved));
    if (!res.ok) throw new Error("복원 실패");
    const data = await res.json();
    if (!data.messages.length) { return resetChat(); }

    currentThreadId = data.thread_id;
    document.getElementById("infoThread").textContent = currentThreadId;
    const linkH = document.getElementById("linkHistory");
    if (linkH) linkH.href = "/api/history/" + encodeURIComponent(currentThreadId);

    const chatBox = document.getElementById("chatBox");
    chatBox.innerHTML = "";
    data.messages.forEach((m) => {
      const isUser = m.type === "human";
      const row = document.createElement("div");
      row.className = "msg-row " + (isUser ? "user" : "bot");
      const contentHtml = isUser ? escapeHtml(m.content) : renderBotReplyContent(m.content, false);
      row.innerHTML = `
        <div class="avatar ${isUser ? "user" : "bot"}">${isUser ? "👤" : "🤖"}</div>
        <div class="bubble">${contentHtml}</div>
      `;
      chatBox.appendChild(row);
    });
    chatBox.scrollTop = chatBox.scrollHeight;

    messageCount = data.messages.length;
    document.getElementById("infoMsgCount").textContent = messageCount + "개 메시지";
    const st = data.token_stats;
    document.getElementById("metricTotalTokens").textContent = st.total_tokens.toLocaleString() + " 토큰";
    document.getElementById("metricTotalCost").textContent = st.total_cost_krw.toFixed(3) + "원 ($" + st.total_cost_usd.toFixed(5) + ")";

    loadSessionsList();
  } catch (e) {
    resetChat();
  }
}

// LangGraph Store 검색 캐시 통계 조회
async function loadCacheStats() {
  try {
    const res = await fetch("/api/cache/stats");
    if (!res.ok) return;
    const data = await res.json();
    const hitEl = document.getElementById("cacheHitCount");
    const missEl = document.getElementById("cacheMissCount");
    const itemEl = document.getElementById("cachedItemCount");
    if (hitEl) hitEl.textContent = data.hits || 0;
    if (missEl) missEl.textContent = data.misses || 0;
    if (itemEl) itemEl.textContent = data.cached_count || 0;
  } catch (e) {}
}

// LangGraph Store 검색 캐시 비우기
async function clearSearchCache() {
  if (!confirm("LangGraph Store에 저장된 구글 검색 캐시를 모두 비우시겠습니까?")) return;
  try {
    const res = await fetch("/api/cache/clear", { method: "POST" });
    if (!res.ok) throw new Error("캐시 삭제 실패");
    const data = await res.json();
    await loadCacheStats();
    alert("✅ " + data.message);
  } catch (e) {
    alert("오류: " + e.message);
  }
}

// 서버 상태 확인
async function loadHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    const stEl = document.getElementById("infoStorage");
    if (data.storage_backend === "postgres") {
      stEl.innerHTML = "<b style='color:#059669;'>PostgreSQL 🟢</b>";
    } else {
      stEl.innerHTML = "<b style='color:#d97706;'>InMemory (임시) 🟡</b>";
    }
    document.getElementById("infoProject").textContent = data.gcp_project;
    document.getElementById("infoLocation").textContent = data.gcp_location;
  } catch (e) {}
}

// 페이지 로드 시 초기화
window.addEventListener("DOMContentLoaded", () => {
  // 저장된 Output length 복원
  const savedLen = localStorage.getItem("chat_output_length");
  if (savedLen) {
    setTokenPreset(parseInt(savedLen, 10));
  } else {
    syncOutputLength("preset");
  }

  // 저장된 모델 및 추론 수준(Reasoning Effort) 복원
  const savedModel = localStorage.getItem("chat_model") || "gemini";
  const savedEffort = localStorage.getItem("chat_reasoning_effort") || "medium";
  const savedLabel = localStorage.getItem("chat_model_label") || (savedModel === "ollama" ? "Ollama" : (savedEffort === "high" ? "사고 모델" : (savedEffort === "low" ? "빠른 추론" : "Flash")));
  selectModelEffort(savedModel, savedEffort, savedLabel);

  // 팝업 메뉴 바깥 클릭 시 닫기
  document.addEventListener("click", (e) => {
    const container = document.getElementById("modelPickerContainer");
    const menu = document.getElementById("modelPopupMenu");
    const caret = document.getElementById("capsuleCaret");
    if (container && menu && !container.contains(e.target)) {
      menu.style.display = "none";
      if (caret) caret.textContent = "▾";
    }
  });

  loadHealth();
  restoreSession();
  loadSessionsList();
  loadCacheStats();
  updateModelUI();
});
