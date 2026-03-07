/* =========================================================
   MarketMinds — Frontend Application Logic  (v0.4)
   =========================================================
   Features:
   - Streaming SSE responses (token-by-token rendering)
   - Interactive stock ticker cards with sparkline charts
   - Chat with markdown rendering (via marked.js)
   - Connection status bar
   - Loading indicator (typing dots)
   - Suggested prompt buttons
   - PDF upload via drag-and-drop or file picker
   - Chat history persistence (localStorage + server sessions)
   - Auto-resize textarea
   ========================================================= */

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------
const STORAGE_KEY = "marketminds_chat_history";
const SESSION_KEY = "marketminds_session_id";
const STREAM_KEY = "marketminds_streaming";
let isWaiting = false;
let sessionId = null;
let streamingEnabled = true;

// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
    // Configure marked.js
    if (typeof marked !== "undefined") {
        marked.setOptions({
            breaks: true,
            gfm: true,
            sanitize: false,
        });
    }

    // Restore or create session ID
    sessionId = localStorage.getItem(SESSION_KEY);
    if (!sessionId) {
        sessionId = crypto.randomUUID ? crypto.randomUUID() : generateUUID();
        localStorage.setItem(SESSION_KEY, sessionId);
    }

    // Restore streaming preference
    const savedStream = localStorage.getItem(STREAM_KEY);
    streamingEnabled = savedStream === null ? true : savedStream === "true";
    updateStreamToggle();

    // Auto-resize textarea
    const textarea = document.getElementById("question");
    textarea.addEventListener("input", autoResizeTextarea);

    // Enter to send (Shift+Enter for newline)
    textarea.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendQuestion();
        }
    });

    // Drag-and-drop for upload
    setupDragAndDrop();

    // File input for upload
    const fileInput = document.getElementById("file-input");
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            uploadFile(e.target.files[0]);
            e.target.value = "";
        }
    });

    // Restore chat history
    restoreChat();

    // Check backend health
    checkHealth();
    setInterval(checkHealth, 30000);
});


// ---------------------------------------------------------------------------
// Session management
// ---------------------------------------------------------------------------
function generateUUID() {
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
        const r = (Math.random() * 16) | 0;
        const v = c === "x" ? r : (r & 0x3) | 0x8;
        return v.toString(16);
    });
}


// ---------------------------------------------------------------------------
// Status bar
// ---------------------------------------------------------------------------
async function checkHealth() {
    const dot = document.getElementById("status-dot");
    const text = document.getElementById("status-text");

    try {
        const res = await fetch("/health");
        const data = await res.json();

        dot.className = "status-dot online";
        text.textContent = `${data.llm_provider} · ${data.vectors_stored} vectors · v${data.version}`;
    } catch {
        dot.className = "status-dot offline";
        text.textContent = "Backend offline";
    }
}


// ---------------------------------------------------------------------------
// Streaming toggle
// ---------------------------------------------------------------------------
function toggleStreaming() {
    streamingEnabled = !streamingEnabled;
    localStorage.setItem(STREAM_KEY, streamingEnabled);
    updateStreamToggle();
}

function updateStreamToggle() {
    const btn = document.getElementById("btn-stream-toggle");
    if (streamingEnabled) {
        btn.classList.add("active");
        btn.title = "Streaming mode ON — click to disable";
    } else {
        btn.classList.remove("active");
        btn.title = "Streaming mode OFF — click to enable";
    }
}


// ---------------------------------------------------------------------------
// Chat — Send & Receive
// ---------------------------------------------------------------------------
async function sendQuestion() {
    if (isWaiting) return;

    const input = document.getElementById("question");
    const question = input.value.trim();
    if (!question) return;

    // Hide welcome screen
    hideWelcome();

    // Show user message
    addMessage(question, "user");
    input.value = "";
    autoResizeTextarea();

    // Show loading indicator
    const loadingEl = showLoading();

    // Disable input while waiting
    setInputState(false);

    try {
        if (streamingEnabled) {
            await sendStreaming(question, loadingEl);
        } else {
            await sendBatch(question, loadingEl);
        }
    } catch (err) {
        removeLoading(loadingEl);
        addMessage("⚠️ Could not reach the backend. Is the server running?", "bot");
        console.error(err);
    } finally {
        setInputState(true);
    }
}

// --- Batch mode (original) ---
async function sendBatch(question, loadingEl) {
    const response = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            question,
            session_id: sessionId,
        }),
    });

    removeLoading(loadingEl);

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const detail = errorData.detail || `Server error ${response.status}`;
        addMessage(`⚠️ ${detail}`, "bot");
        return;
    }

    const data = await response.json();

    if (data.session_id && data.session_id !== sessionId) {
        sessionId = data.session_id;
        localStorage.setItem(SESSION_KEY, sessionId);
    }

    addMessage(data.answer, "bot");
}

// --- Streaming mode (SSE) ---
async function sendStreaming(question, loadingEl) {
    const response = await fetch("/ask/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            question,
            session_id: sessionId,
        }),
    });

    if (!response.ok) {
        removeLoading(loadingEl);
        const errorData = await response.json().catch(() => ({}));
        addMessage(`⚠️ ${errorData.detail || `Server error ${response.status}`}`, "bot");
        return;
    }

    // Remove loading dots and create a streaming message
    removeLoading(loadingEl);
    const streamDiv = createStreamingMessage();

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let accumulated = "";

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n");

        for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const jsonStr = line.slice(6).trim();
            if (!jsonStr) continue;

            try {
                const data = JSON.parse(jsonStr);

                // Update session ID
                if (data.session_id && data.session_id !== sessionId) {
                    sessionId = data.session_id;
                    localStorage.setItem(SESSION_KEY, sessionId);
                }

                if (data.done) {
                    // Finalize the message
                    finalizeStreamingMessage(streamDiv, accumulated);
                    break;
                }

                if (data.token) {
                    accumulated += data.token;
                    // Render progressively
                    streamDiv.innerHTML = renderMarkdown(accumulated);
                    scrollToBottom();
                }
            } catch {
                // Skip malformed JSON
            }
        }
    }

    // Safety: finalize if not already done
    if (accumulated && streamDiv.getAttribute("data-finalized") !== "true") {
        finalizeStreamingMessage(streamDiv, accumulated);
    }

    saveChat();
}

function createStreamingMessage() {
    const chat = document.getElementById("chat");
    const div = document.createElement("div");
    div.className = "message bot streaming";
    div.setAttribute("data-role", "bot");
    div.innerHTML = '<span class="cursor-blink">▊</span>';
    chat.appendChild(div);
    scrollToBottom();
    return div;
}

function finalizeStreamingMessage(div, text) {
    div.classList.remove("streaming");
    div.innerHTML = renderMarkdown(text);
    div.setAttribute("data-finalized", "true");

    // Detect ticker mentions and add interactive cards
    detectAndAddTickerCards(div, text);

    scrollToBottom();
    saveChat();
}


// ---------------------------------------------------------------------------
// Ticker card detection & rendering
// ---------------------------------------------------------------------------
const TICKER_PATTERN = /\*\*([A-Z]{2,5}(?:\.[A-Z]{1,2})?)\*\*/g;

function detectAndAddTickerCards(messageDiv, text) {
    const tickers = new Set();
    let match;
    while ((match = TICKER_PATTERN.exec(text)) !== null) {
        tickers.add(match[1]);
    }

    for (const ticker of tickers) {
        fetchTickerCard(ticker, messageDiv);
    }
}

async function fetchTickerCard(ticker, parentDiv) {
    try {
        const res = await fetch(`/ticker/${ticker}`);
        if (!res.ok) return;
        const data = await res.json();

        const card = document.createElement("div");
        card.className = "ticker-card";

        const isPositive = data.change_pct >= 0;
        const arrow = isPositive ? "▲" : "▼";
        const colorClass = isPositive ? "positive" : "negative";

        card.innerHTML = `
            <div class="ticker-card-header">
                <span class="ticker-symbol">${data.ticker}</span>
                <span class="ticker-price">${data.price} ${data.currency}</span>
                <span class="ticker-change ${colorClass}">${arrow} ${Math.abs(data.change_pct).toFixed(2)}%</span>
            </div>
            <canvas class="sparkline-canvas" width="160" height="40"></canvas>
            <div class="ticker-card-actions">
                <button onclick="fetchAndShowNews('${data.ticker}')" class="btn-ticker-action">📰 News</button>
                <button onclick="askAboutTicker('${data.ticker}')" class="btn-ticker-action">💬 Ask</button>
            </div>
        `;

        parentDiv.appendChild(card);

        // Draw sparkline
        if (data.sparkline && data.sparkline.length > 0) {
            const canvas = card.querySelector(".sparkline-canvas");
            drawSparkline(canvas, data.sparkline, isPositive);
        }
    } catch {
        // Silently ignore ticker card errors
    }
}

function drawSparkline(canvas, data, isPositive) {
    if (typeof Chart === "undefined") return;

    const ctx = canvas.getContext("2d");
    const gradient = ctx.createLinearGradient(0, 0, 0, 40);
    const color = isPositive ? "rgba(52, 211, 153, " : "rgba(248, 113, 113, ";
    gradient.addColorStop(0, color + "0.3)");
    gradient.addColorStop(1, color + "0.0)");

    new Chart(ctx, {
        type: "line",
        data: {
            labels: data.map((_, i) => i),
            datasets: [{
                data: data,
                borderColor: isPositive ? "#34d399" : "#f87171",
                borderWidth: 1.5,
                fill: true,
                backgroundColor: gradient,
                pointRadius: 0,
                tension: 0.4,
            }],
        },
        options: {
            responsive: false,
            plugins: { legend: { display: false }, tooltip: { enabled: false } },
            scales: {
                x: { display: false },
                y: { display: false },
            },
            animation: { duration: 600 },
        },
    });
}

function askAboutTicker(ticker) {
    const input = document.getElementById("question");
    input.value = `Tell me about ${ticker} — what are the key fundamentals and recent performance?`;
    sendQuestion();
}

async function fetchAndShowNews(ticker) {
    const chat = document.getElementById("chat");

    // Show loading
    const newsLoading = document.createElement("div");
    newsLoading.className = "message bot";
    newsLoading.setAttribute("data-role", "bot");
    newsLoading.innerHTML = `<em>Fetching news for ${ticker}…</em>`;
    chat.appendChild(newsLoading);
    scrollToBottom();

    try {
        const res = await fetch(`/news/${ticker}`);
        const data = await res.json();

        if (!res.ok || !data.news || data.news.length === 0) {
            newsLoading.innerHTML = `<p>No recent news found for <strong>${ticker}</strong>.</p>`;
            return;
        }

        let html = `<div class="news-panel"><h4>📰 Latest News — ${ticker}</h4><ul class="news-list">`;
        for (const item of data.news) {
            html += `<li>
                <a href="${item.link}" target="_blank" rel="noopener">${item.title}</a>
                <span class="news-source">${item.publisher}</span>
            </li>`;
        }
        html += "</ul></div>";
        newsLoading.innerHTML = html;
    } catch {
        newsLoading.innerHTML = `<p>⚠️ Could not fetch news for ${ticker}.</p>`;
    }

    scrollToBottom();
    saveChat();
}


// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------
function addMessage(text, role) {
    const chat = document.getElementById("chat");
    const div = document.createElement("div");
    div.className = `message ${role}`;
    div.setAttribute("data-role", role);

    if (role === "bot") {
        div.innerHTML = renderMarkdown(text);
        // Check for ticker mentions
        detectAndAddTickerCards(div, text);
    } else {
        div.textContent = text;
    }

    chat.appendChild(div);
    scrollToBottom();
    saveChat();
}

function renderMarkdown(text) {
    if (typeof marked !== "undefined" && marked.parse) {
        try {
            return marked.parse(text);
        } catch {
            // Fallback
        }
    }
    return text.replace(/\n/g, "<br>");
}

function scrollToBottom() {
    const chat = document.getElementById("chat");
    requestAnimationFrame(() => {
        chat.scrollTop = chat.scrollHeight;
    });
}


// ---------------------------------------------------------------------------
// Loading indicator
// ---------------------------------------------------------------------------
function showLoading() {
    const chat = document.getElementById("chat");
    const div = document.createElement("div");
    div.className = "message bot loading";
    div.id = "loading-indicator";
    div.innerHTML = `
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
    `;
    chat.appendChild(div);
    scrollToBottom();
    return div;
}

function removeLoading(el) {
    if (el && el.parentNode) {
        el.parentNode.removeChild(el);
    }
}

function setInputState(enabled) {
    isWaiting = !enabled;
    const textarea = document.getElementById("question");
    const btn = document.getElementById("btn-send");
    textarea.disabled = !enabled;
    btn.disabled = !enabled;

    if (enabled) {
        textarea.focus();
    }
}


// ---------------------------------------------------------------------------
// Welcome screen & suggestions
// ---------------------------------------------------------------------------
function hideWelcome() {
    const welcome = document.getElementById("welcome");
    if (welcome) {
        welcome.style.display = "none";
    }
}

function useSuggestion(btn) {
    // Get text content excluding the icon span
    const icon = btn.querySelector(".suggestion-icon");
    const question = icon
        ? btn.textContent.replace(icon.textContent, "").trim()
        : btn.textContent.trim();
    document.getElementById("question").value = question;
    sendQuestion();
}


// ---------------------------------------------------------------------------
// Auto-resize textarea
// ---------------------------------------------------------------------------
function autoResizeTextarea() {
    const textarea = document.getElementById("question");
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + "px";
}


// ---------------------------------------------------------------------------
// File Upload
// ---------------------------------------------------------------------------
function toggleUploadPanel() {
    const panel = document.getElementById("upload-panel");
    panel.classList.toggle("hidden");
}

function setupDragAndDrop() {
    const dropzone = document.getElementById("dropzone");
    if (!dropzone) return;

    ["dragenter", "dragover"].forEach((event) => {
        dropzone.addEventListener(event, (e) => {
            e.preventDefault();
            dropzone.classList.add("drag-over");
        });
    });

    ["dragleave", "drop"].forEach((event) => {
        dropzone.addEventListener(event, (e) => {
            e.preventDefault();
            dropzone.classList.remove("drag-over");
        });
    });

    dropzone.addEventListener("drop", (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            uploadFile(files[0]);
        }
    });
}

async function uploadFile(file) {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
        showUploadStatus("Only PDF files are supported.", "error");
        return;
    }

    if (file.size > 50 * 1024 * 1024) {
        showUploadStatus("File too large. Maximum size is 50 MB.", "error");
        return;
    }

    showUploadStatus(`Uploading "${file.name}"…`, "loading");

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("/upload", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (response.ok) {
            showUploadStatus(
                `✅ "${data.filename}" ingested — ${data.chunks} searchable chunks created.`,
                "success"
            );
            // Refresh health to update vector count
            checkHealth();
            setTimeout(() => {
                const status = document.getElementById("upload-status");
                status.classList.add("hidden");
            }, 5000);
        } else {
            showUploadStatus(`❌ Upload failed: ${data.detail || "Unknown error"}`, "error");
        }
    } catch (err) {
        showUploadStatus("❌ Upload failed: Could not reach server.", "error");
        console.error(err);
    }
}

function showUploadStatus(message, type) {
    const status = document.getElementById("upload-status");
    status.textContent = message;
    status.className = `upload-status ${type}`;
}


// ---------------------------------------------------------------------------
// Chat persistence (localStorage) + session management
// ---------------------------------------------------------------------------
function saveChat() {
    const chat = document.getElementById("chat");
    const messages = [];

    chat.querySelectorAll(".message").forEach((el) => {
        if (el.classList.contains("loading")) return; // skip loading indicators
        const role = el.getAttribute("data-role") || (el.classList.contains("user") ? "user" : "bot");
        const content = role === "user" ? el.textContent : el.innerHTML;
        messages.push({ role, content });
    });

    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
    } catch {
        // localStorage might be full — silently ignore
    }
}

function restoreChat() {
    try {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (!saved) return;

        const messages = JSON.parse(saved);
        if (!Array.isArray(messages) || messages.length === 0) return;

        hideWelcome();

        const chat = document.getElementById("chat");
        messages.forEach(({ role, content }) => {
            const div = document.createElement("div");
            div.className = `message ${role}`;
            div.setAttribute("data-role", role);

            if (role === "user") {
                div.textContent = content;
            } else {
                div.innerHTML = content;
            }

            chat.appendChild(div);
        });

        scrollToBottom();
    } catch {
        localStorage.removeItem(STORAGE_KEY);
    }
}

function clearChat() {
    const chat = document.getElementById("chat");

    // Remove only messages (keep welcome)
    chat.querySelectorAll(".message").forEach((el) => el.remove());

    // Show welcome again
    const welcome = document.getElementById("welcome");
    if (welcome) {
        welcome.style.display = "";
    }

    // Clear localStorage
    localStorage.removeItem(STORAGE_KEY);

    // Clear server-side session history
    if (sessionId) {
        fetch(`/history/${sessionId}`, { method: "DELETE" }).catch(() => { });
    }

    // Start a new session
    sessionId = crypto.randomUUID ? crypto.randomUUID() : generateUUID();
    localStorage.setItem(SESSION_KEY, sessionId);
}
