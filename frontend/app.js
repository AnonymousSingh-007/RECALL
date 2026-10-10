// ===================================================================
// RECALL Frontend — Mic Recording → /query → Answer Display
// Uses MediaRecorder API to capture audio, POST to backend.
// ===================================================================

const API_BASE = "";  // same-origin; change to "http://localhost:8000" for dev

// --- DOM refs ---
const micBtn         = document.getElementById("mic-btn");
const micIcon        = document.getElementById("mic-icon");
const micWaves       = document.getElementById("mic-waves");
const micLabel       = document.getElementById("mic-label");
const transcriptBox  = document.getElementById("transcript-box");
const transcriptText = document.getElementById("transcript-text");
const loading        = document.getElementById("loading");
const answerBox      = document.getElementById("answer-box");
const answerText     = document.getElementById("answer-text");
const strategyBadge  = document.getElementById("strategy-badge");
const confidenceBadge = document.getElementById("confidence-badge");
const groundedBadge  = document.getElementById("grounded-badge");
const totalLatency   = document.getElementById("total-latency");
const sourcesCount   = document.getElementById("sources-count");
const sourcesList    = document.getElementById("sources-list");
const errorBox       = document.getElementById("error-box");
const errorText      = document.getElementById("error-text");

// Latency elements
const latEls = {
  stt:       document.getElementById("lat-stt"),
  classify:  document.getElementById("lat-classify"),
  retrieve:  document.getElementById("lat-retrieve"),
  rerank:    document.getElementById("lat-rerank"),
  guardrail: document.getElementById("lat-guardrail"),
  generate:  document.getElementById("lat-generate"),
};

// --- State ---
let mediaRecorder = null;
let audioChunks   = [];
let isRecording   = false;

// --- Helpers ---
function show(el) { el.classList.remove("hidden"); }
function hide(el) { el.classList.add("hidden"); }

function resetUI() {
  hide(answerBox);
  hide(errorBox);
  hide(loading);
  hide(transcriptBox);
}

function fmtMs(val) {
  if (val == null) return "—";
  return `${Math.round(val)}ms`;
}

// --- Recording ---
async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: 16000,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      }
    });

    // Prefer WAV-compatible format, fall back gracefully
    const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? "audio/webm;codecs=opus"
      : "audio/webm";

    mediaRecorder = new MediaRecorder(stream, { mimeType });
    audioChunks = [];

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(audioChunks, { type: mimeType });
      submitAudio(blob);
    };

    mediaRecorder.start();
    isRecording = true;

    micBtn.classList.add("recording");
    micIcon.style.opacity = "0";
    show(micWaves);
    micLabel.textContent = "Tap to stop recording";

  } catch (err) {
    showError(`Microphone access denied: ${err.message}`);
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
  isRecording = false;

  micBtn.classList.remove("recording");
  micIcon.style.opacity = "1";
  hide(micWaves);
  micLabel.textContent = "Processing…";
}

// --- Submit Audio ---
async function submitAudio(blob) {
  resetUI();
  show(loading);

  const formData = new FormData();
  formData.append("file", blob, "recording.webm");

  try {
    const res = await fetch(`${API_BASE}/query`, {
      method: "POST",
      body: formData,
    });

    hide(loading);

    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `Server error (${res.status})`);
    }

    const data = await res.json();
    renderResponse(data);

  } catch (err) {
    hide(loading);
    showError(err.message);
  }
}

// --- Render Response ---
function renderResponse(data) {
  // Transcript
  if (data.query_text) {
    transcriptText.textContent = data.query_text;
    show(transcriptBox);
  }

  // Answer
  answerText.textContent = data.answer || "No answer available.";

  // Strategy badges
  const strategies = data.strategy_used || [];
  strategyBadge.textContent = strategies.length
    ? `🔍 ${strategies.join(" + ")}`
    : "";

  // Confidence
  const conf = data.confidence;
  if (conf != null) {
    const pct = (conf * 100).toFixed(1);
    confidenceBadge.textContent = `📊 ${pct}% confidence`;
    confidenceBadge.classList.toggle("low", conf < 0.05);
  } else {
    confidenceBadge.textContent = "";
  }

  // Grounded
  if (data.grounded != null) {
    groundedBadge.textContent = data.grounded ? "✅ Grounded" : "⚠️ Ungrounded";
    groundedBadge.className = `badge grounded-badge ${data.grounded ? "yes" : "no"}`;
  } else {
    groundedBadge.textContent = "";
  }

  // Latency breakdown
  const lb = data.latency_breakdown || {};
  totalLatency.textContent = `Total: ${fmtMs(lb.total_ms)}`;
  latEls.stt.textContent       = fmtMs(lb.stt_ms);
  latEls.classify.textContent  = fmtMs(lb.classify_ms);
  latEls.retrieve.textContent  = fmtMs(lb.retrieve_ms);
  latEls.rerank.textContent    = fmtMs(lb.rerank_ms);
  latEls.guardrail.textContent = fmtMs(lb.guardrail_ms);
  latEls.generate.textContent  = fmtMs(lb.generate_ms);

  // Sources
  const sources = data.sources || [];
  sourcesCount.textContent = `Sources (${sources.length})`;
  sourcesList.innerHTML = "";

  sources.forEach((src, i) => {
    const item = document.createElement("div");
    item.className = "source-item";
    item.innerHTML = `
      <div class="source-meta">
        <span class="badge strategy-badge">🔍 ${src.strategy || "—"}</span>
        <span class="badge confidence-badge">Score: ${(src.score || 0).toFixed(4)}</span>
      </div>
      <p class="source-text">${escapeHtml(src.text || "")}</p>
    `;
    sourcesList.appendChild(item);
  });

  show(answerBox);
  micLabel.textContent = "Tap to ask another question";
}

function showError(msg) {
  errorText.textContent = msg;
  show(errorBox);
  micLabel.textContent = "Tap to try again";
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// --- Event Listener ---
micBtn.addEventListener("click", () => {
  if (isRecording) {
    stopRecording();
  } else {
    resetUI();
    startRecording();
  }
});