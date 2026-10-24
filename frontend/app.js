// ===================================================================
// RECALL Frontend — Premium UI v2
// Mic Recording → /query → Answer Display
// 3D Orb Visualization · Particle System · Micro-animations
// ===================================================================

const API_BASE = "";  // same-origin; change to "http://localhost:8000" for dev

// --- DOM refs ---
const micBtn         = document.getElementById("mic-btn");
const micIcon        = document.getElementById("mic-icon");
const stopIconEl     = document.getElementById("stop-icon");
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
const orbCanvas      = document.getElementById("orb-canvas");
const particlesEl    = document.getElementById("particles");
const textForm       = document.getElementById("text-form");
const textInput      = document.getElementById("text-input");

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

// ===================================================================
// 3D ORB VISUALIZATION
// A WebGL-free canvas orb that pulses/ripples when recording.
// ===================================================================
const ctx = orbCanvas.getContext("2d");
let orbAnimId = null;
let orbTime = 0;
let orbState = "idle"; // "idle" | "recording" | "processing"

function getOrbSize() {
  const rect = orbCanvas.getBoundingClientRect();
  return { w: rect.width, h: rect.height, cx: rect.width / 2, cy: rect.height / 2 };
}

function resizeOrbCanvas() {
  const container = document.getElementById("orb-container");
  const size = container.getBoundingClientRect();
  orbCanvas.width = size.width * window.devicePixelRatio;
  orbCanvas.height = size.height * window.devicePixelRatio;
  orbCanvas.style.width = size.width + "px";
  orbCanvas.style.height = size.height + "px";
  ctx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
}

function drawOrb(timestamp) {
  orbTime = timestamp * 0.001; // seconds
  const { w, h } = getOrbSize();
  const cx = w / 2;
  const cy = h / 2;

  ctx.clearRect(0, 0, w, h);

  if (orbState === "idle") {
    drawIdleOrb(cx, cy, w);
  } else if (orbState === "recording") {
    drawRecordingOrb(cx, cy, w);
  } else if (orbState === "processing") {
    drawProcessingOrb(cx, cy, w);
  }

  orbAnimId = requestAnimationFrame(drawOrb);
}

function drawIdleOrb(cx, cy, size) {
  const baseRadius = size * 0.3;

  // Outer glow ring
  const glowGrad = ctx.createRadialGradient(cx, cy, baseRadius * 0.6, cx, cy, baseRadius * 1.4);
  glowGrad.addColorStop(0, "rgba(56, 132, 255, 0.0)");
  glowGrad.addColorStop(0.6, "rgba(56, 132, 255, 0.03)");
  glowGrad.addColorStop(1, "rgba(56, 132, 255, 0.0)");
  ctx.fillStyle = glowGrad;
  ctx.fillRect(0, 0, size, size);

  // Orbiting dots
  const dotCount = 40;
  for (let i = 0; i < dotCount; i++) {
    const angle = (i / dotCount) * Math.PI * 2 + orbTime * 0.3;
    const wobble = Math.sin(orbTime * 0.8 + i * 0.5) * 4;
    const r = baseRadius + 10 + wobble;
    const x = cx + Math.cos(angle) * r;
    const y = cy + Math.sin(angle) * r;
    const alpha = 0.15 + Math.sin(orbTime + i * 0.4) * 0.1;
    const dotSize = 1.2 + Math.sin(orbTime * 0.5 + i) * 0.5;

    ctx.beginPath();
    ctx.arc(x, y, dotSize, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(96, 165, 250, ${alpha})`;
    ctx.fill();
  }

  // Inner ring
  ctx.beginPath();
  ctx.arc(cx, cy, baseRadius - 2, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(56, 132, 255, 0.06)";
  ctx.lineWidth = 1;
  ctx.stroke();
}

function drawRecordingOrb(cx, cy, size) {
  const baseRadius = size * 0.3;
  const time = orbTime;

  // Pulsing outer glow
  const pulseScale = 1 + Math.sin(time * 3) * 0.08;
  const glowRadius = baseRadius * 1.5 * pulseScale;

  const glowGrad = ctx.createRadialGradient(cx, cy, baseRadius * 0.3, cx, cy, glowRadius);
  glowGrad.addColorStop(0, "rgba(248, 113, 113, 0.08)");
  glowGrad.addColorStop(0.5, "rgba(248, 113, 113, 0.04)");
  glowGrad.addColorStop(1, "rgba(248, 113, 113, 0.0)");
  ctx.fillStyle = glowGrad;
  ctx.fillRect(0, 0, size, size);

  // Waveform ring — organic, flowing shape
  const wavePoints = 72;
  ctx.beginPath();
  for (let i = 0; i <= wavePoints; i++) {
    const angle = (i / wavePoints) * Math.PI * 2;
    const wave1 = Math.sin(angle * 3 + time * 4) * 8;
    const wave2 = Math.sin(angle * 5 - time * 2.5) * 5;
    const wave3 = Math.sin(angle * 7 + time * 6) * 3;
    const r = baseRadius + wave1 + wave2 + wave3;
    const x = cx + Math.cos(angle) * r;
    const y = cy + Math.sin(angle) * r;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.strokeStyle = "rgba(248, 113, 113, 0.4)";
  ctx.lineWidth = 2;
  ctx.stroke();

  // Second waveform ring (offset)
  ctx.beginPath();
  for (let i = 0; i <= wavePoints; i++) {
    const angle = (i / wavePoints) * Math.PI * 2;
    const wave1 = Math.sin(angle * 4 - time * 3) * 6;
    const wave2 = Math.sin(angle * 6 + time * 5) * 4;
    const r = baseRadius + 16 + wave1 + wave2;
    const x = cx + Math.cos(angle) * r;
    const y = cy + Math.sin(angle) * r;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.strokeStyle = "rgba(248, 113, 113, 0.15)";
  ctx.lineWidth = 1;
  ctx.stroke();

  // Orbiting particles that speed up
  const particleCount = 60;
  for (let i = 0; i < particleCount; i++) {
    const angle = (i / particleCount) * Math.PI * 2 + time * 1.5;
    const wobble = Math.sin(time * 3 + i * 0.7) * 10;
    const r = baseRadius + wobble;
    const x = cx + Math.cos(angle) * r;
    const y = cy + Math.sin(angle) * r;
    const alpha = 0.2 + Math.sin(time * 2 + i * 0.5) * 0.15;
    const dotSize = 1 + Math.sin(time * 4 + i) * 0.8;

    ctx.beginPath();
    ctx.arc(x, y, dotSize, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(248, 113, 113, ${alpha})`;
    ctx.fill();
  }
}

function drawProcessingOrb(cx, cy, size) {
  const baseRadius = size * 0.3;
  const time = orbTime;

  // Spinning gradient arc
  const startAngle = time * 2;
  const arcLength = Math.PI * 1.2;

  ctx.beginPath();
  ctx.arc(cx, cy, baseRadius + 8, startAngle, startAngle + arcLength);
  const arcGrad = ctx.createLinearGradient(
    cx + Math.cos(startAngle) * baseRadius,
    cy + Math.sin(startAngle) * baseRadius,
    cx + Math.cos(startAngle + arcLength) * baseRadius,
    cy + Math.sin(startAngle + arcLength) * baseRadius
  );
  arcGrad.addColorStop(0, "rgba(56, 132, 255, 0.0)");
  arcGrad.addColorStop(0.5, "rgba(56, 132, 255, 0.5)");
  arcGrad.addColorStop(1, "rgba(108, 92, 231, 0.0)");
  ctx.strokeStyle = arcGrad;
  ctx.lineWidth = 2.5;
  ctx.lineCap = "round";
  ctx.stroke();

  // Inner dots orbiting slowly
  const dotCount = 30;
  for (let i = 0; i < dotCount; i++) {
    const angle = (i / dotCount) * Math.PI * 2 + time * 0.5;
    const r = baseRadius + 2 + Math.sin(time + i * 0.3) * 3;
    const x = cx + Math.cos(angle) * r;
    const y = cy + Math.sin(angle) * r;
    const alpha = 0.1 + Math.sin(time * 1.5 + i * 0.6) * 0.08;

    ctx.beginPath();
    ctx.arc(x, y, 1, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(96, 165, 250, ${alpha})`;
    ctx.fill();
  }
}

// Initialize orb
resizeOrbCanvas();
window.addEventListener("resize", resizeOrbCanvas);
orbAnimId = requestAnimationFrame(drawOrb);

// ===================================================================
// PARTICLE SYSTEM — ambient floating particles
// ===================================================================
function spawnParticles() {
  const count = 18;
  for (let i = 0; i < count; i++) {
    const p = document.createElement("div");
    p.className = "particle";
    p.style.left = Math.random() * 100 + "%";
    p.style.top = (60 + Math.random() * 40) + "%";
    p.style.animationDuration = (8 + Math.random() * 12) + "s";
    p.style.animationDelay = (Math.random() * 10) + "s";
    p.style.width = (1 + Math.random() * 1.5) + "px";
    p.style.height = p.style.width;
    particlesEl.appendChild(p);
  }
}
spawnParticles();

// ===================================================================
// RECORDING
// ===================================================================
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
    orbState = "recording";

    micBtn.classList.add("recording");
    hide(micIcon);
    show(stopIconEl);
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
  orbState = "processing";

  micBtn.classList.remove("recording");
  show(micIcon);
  hide(stopIconEl);
  micLabel.textContent = "Processing…";
}

// ===================================================================
// SUBMIT AUDIO
// ===================================================================
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
    orbState = "idle";
    renderResponse(data);

  } catch (err) {
    hide(loading);
    orbState = "idle";
    showError(err.message);
  }
}

async function submitText(query) {
  if (!query || !query.trim()) return;
  resetUI();
  show(loading);
  orbState = "processing";
  transcriptText.textContent = query;
  show(transcriptBox);

  try {
    const res = await fetch(`${API_BASE}/query-text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: query.trim() }),
    });

    hide(loading);

    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `Server error (${res.status})`);
    }

    const data = await res.json();
    orbState = "idle";
    renderResponse(data);

  } catch (err) {
    hide(loading);
    orbState = "idle";
    showError(err.message);
  }
}


// ===================================================================
// RENDER RESPONSE
// ===================================================================
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

// --- Event Listeners ---
micBtn.addEventListener("click", () => {
  if (isRecording) {
    stopRecording();
  } else {
    resetUI();
    startRecording();
  }
});

if (textForm) {
  textForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const query = textInput.value.trim();
    if (!query) return;
    textInput.value = "";
    submitText(query);
  });
}