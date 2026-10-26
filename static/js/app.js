let currentThreadId = "";
let messageCount = 0;

// 세션 초기화 (새 대화 시작)
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

    // 세션 목록 갱신
    loadSessionsList();
  } catch (e) {
    alert("세션 초기화 실패: " + e);
  }
}

// 시간 포맷 변환 (MM/DD HH:mm)
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

// 대화 세션 목록 불러오기 (사이드바)
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

      const title = escapeHtml(s.title || `대화 (${s.thread_id})`);
      const timeStr = formatTime(s.updated_at);
      const totalTok = (s.total_input || 0) + (s.total_output || 0);
      const tokStr = totalTok > 0 ? `${totalTok.toLocaleString()}T` : "";
      const costStr = s.total_cost_krw > 0 ? `${s.total_cost_krw.toFixed(2)}원` : "";
      const metaParts = [timeStr, tokStr, costStr].filter(Boolean);
      const metaStr = metaParts.length > 0 ? metaParts.join(" · ") : "대화 진행 전";

      item.innerHTML = `
        <div class="session-item-content">
          <div class="session-item-title" title="${title}">💬 ${title}</div>
          <div class="session-item-meta">
            <span><code>${escapeHtml(s.thread_id)}</code></span>
            <span>${escapeHtml(metaStr)}</span>
          </div>
        </div>
        <button class="btn-del-session" title="대화 삭제" onclick="deleteSession(event, '${escapeHtml(s.thread_id)}')">🗑️</button>
      `;
      container.appendChild(item);
    });
  } catch (e) {
    console.error("세션 목록 로드 실패:", e);
  }
}

// 다른 대화 세션으로 전환
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

// 세션 삭제
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

// 모델/검색 UI 상태 업데이트
function updateModelUI() {
  const model = document.querySelector('input[name="model"]:checked').value;
  const mode = document.querySelector('input[name="searchMode"]:checked').value;

  const badgeModel = document.getElementById("badgeModel");
  const badgeSearch = document.getElementById("badgeSearch");
  const searchHint = document.getElementById("searchHint");

  badgeModel.textContent = model === "gemini" ? "Gemini 2.5 Flash" : "Ollama (gemma4:e2b)";

  if (mode === "grounding") {
    badgeSearch.textContent = "🌐 Grounding";
    badgeSearch.className = "badge search-on";
    searchHint.innerHTML = model === "gemini"
      ? "🟢 <b>Vertex AI Grounding</b>: Gemini가 구글 검색을 직접 수행해 답변합니다."
      : "⚠️ <b>알림</b>: Grounding은 Gemini 전용입니다. Ollama에서는 적용되지 않습니다.";
  } else if (mode === "cse") {
    badgeSearch.textContent = "🔎 Custom Search";
    badgeSearch.className = "badge search-on";
    searchHint.innerHTML = "🟢 <b>Custom Search</b>: google_cse_search 도구로 Programmable Search Engine을 호출합니다. (.env의 GOOGLE_CSE_API_KEY / GOOGLE_CSE_ID 필요)";
  } else {
    badgeSearch.textContent = "검색 OFF";
    badgeSearch.className = "badge";
    searchHint.innerHTML = "⚪ <b>검색 OFF</b>: 모델 자체 학습 지식으로 답변합니다.";
  }
}

// 메시지 전송
async function sendMessage() {
  const input = document.getElementById("userInput");
  const msg = input.value.trim();
  if (!msg) return;

  const model = document.querySelector('input[name="model"]:checked').value;
  const searchMode = document.querySelector('input[name="searchMode"]:checked').value;
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
      }),
    });

    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || "서버 응답 오류");
    }

    const data = await res.json();
    currentThreadId = data.thread_id;
    localStorage.setItem("chat_thread_id", currentThreadId);
    document.getElementById("infoThread").textContent = currentThreadId;
    const linkH = document.getElementById("linkHistory");
    if (linkH) linkH.href = "/api/history/" + encodeURIComponent(currentThreadId);

    // 로딩 제거
    loadingRow.remove();

    // 봇 메시지 렌더링
    let sourcesHtml = "";
    if (data.sources && data.sources.length > 0) {
      sourcesHtml = '<div class="sources-box"><b>🌐 실시간 검색 출처:</b><br>';
      data.sources.forEach((s) => {
        sourcesHtml += `<a class="source-link" href="${s.uri}" target="_blank" rel="noopener">🔗 ${escapeHtml(s.title)}</a>`;
      });
      sourcesHtml += '</div>';
    }

    const botRow = document.createElement("div");
    botRow.className = "msg-row bot";
    botRow.innerHTML = `
      <div class="avatar bot">🤖</div>
      <div class="bubble">
        ${escapeHtml(data.reply)}
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

    // 메모리 메시지 카운트 갱신
    messageCount += 2;
    document.getElementById("infoMsgCount").textContent = messageCount + "개 메시지";

    // 사이드바 세션 목록 갱신
    loadSessionsList();

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

// HTML 특수문자 이스케이프
function escapeHtml(str) {
  if (!str) return "";
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// 초기 실행: 저장된 세션(thread_id)이 있으면 DB에서 대화 내용을 복원, 없으면 새 세션 시작
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
      row.innerHTML = `
        <div class="avatar ${isUser ? "user" : "bot"}">${isUser ? "👤" : "🤖"}</div>
        <div class="bubble">${escapeHtml(m.content)}</div>
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

// 시스템 상태 확인
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

window.addEventListener("DOMContentLoaded", () => {
  loadHealth();
  restoreSession();
  loadSessionsList();
  updateModelUI();
});

