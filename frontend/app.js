const micBtn = document.getElementById("mic-btn");
const answerBox = document.getElementById("answer-box");
const answerText = document.getElementById("answer-text");
const strategyBadge = document.getElementById("strategy-badge");
const latencyBadge = document.getElementById("latency-badge");

micBtn.addEventListener("click", async () => {
  // TODO Day 4: MediaRecorder -> record audio -> POST to /query
  const res = await fetch("/query", { method: "POST" });
  const data = await res.json();

  answerText.textContent = data.answer ?? "…";
  strategyBadge.textContent = `Strategy: ${(data.strategy_used || []).join(" + ")}`;
  latencyBadge.textContent = `Latency: ${data.latency_breakdown?.total_ms ?? "—"}ms`;
  answerBox.classList.remove("hidden");
});