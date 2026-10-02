"""Run N queries through the full pipeline, report P50/P70/P100.

Currently uses a mock harness that simulates realistic latency ranges.
Swap the `_mock_harness_call` with the real `RAGHarness.run()` once
faiss_store.py and hybrid.py are wired up.

Usage:
    python benchmarks/run_latency_test.py                    # default 150 queries
    python benchmarks/run_latency_test.py --n 50             # 50 queries
    python benchmarks/run_latency_test.py --real              # use real harness (requires indices)
"""

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path so we can import backend modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Sample queries for benchmarking
# ---------------------------------------------------------------------------

SAMPLE_QUERIES = [
    "Who led the Manhattan Project?",
    "What is photosynthesis?",
    "When did World War II end?",
    "Explain how a combustion engine works",
    "What is the capital of France?",
    "How does the immune system fight infections?",
    "Who discovered penicillin?",
    "What causes tides in the ocean?",
    "Describe the process of mitosis",
    "What is the speed of light?",
    "How do solar panels generate electricity?",
    "Who wrote Romeo and Juliet?",
    "What is the theory of relativity?",
    "How are diamonds formed?",
    "What is the largest planet in our solar system?",
    "Explain the water cycle",
    "Who painted the Mona Lisa?",
    "What is DNA?",
    "How do vaccines work?",
    "What causes earthquakes?",
    "Who was the first person to walk on the moon?",
    "What is the periodic table?",
    "How does a computer processor work?",
    "What is the greenhouse effect?",
    "Who invented the telephone?",
    "What is the difference between weather and climate?",
    "How do airplanes fly?",
    "What is the human genome?",
    "Who discovered gravity?",
    "What is quantum mechanics?",
]


# ---------------------------------------------------------------------------
# Mock harness (simulates pipeline latencies)
# ---------------------------------------------------------------------------

def _mock_latency():
    """Generate realistic latency breakdown matching pipeline stages."""
    stt_ms       = random.uniform(200, 600)
    classify_ms  = random.uniform(1, 5)
    retrieve_ms  = random.uniform(50, 200)
    rerank_ms    = random.uniform(30, 120)
    guardrail_ms = random.uniform(1, 10)
    generate_ms  = random.uniform(100, 400)
    total_ms     = stt_ms + classify_ms + retrieve_ms + rerank_ms + guardrail_ms + generate_ms

    return {
        "stt_ms": round(stt_ms, 2),
        "classify_ms": round(classify_ms, 2),
        "retrieve_ms": round(retrieve_ms, 2),
        "rerank_ms": round(rerank_ms, 2),
        "guardrail_ms": round(guardrail_ms, 2),
        "generate_ms": round(generate_ms, 2),
        "total_ms": round(total_ms, 2),
    }


async def _mock_harness_call(query: str) -> dict:
    """Simulate a full pipeline call with realistic latency."""
    latency = _mock_latency()
    # Simulate actual wall-clock delay (scaled down for speed)
    await asyncio.sleep(latency["total_ms"] / 10_000)  # 10x faster than real
    return {
        "query": query,
        "answer": f"[MOCK] Answer for: {query[:50]}",
        "latency_breakdown": latency,
        "strategy_used": random.choice([
            ["sliding_window"],
            ["semantic", "hierarchical"],
            ["metadata_aware"],
            ["semantic", "sliding_window"],
        ]),
        "grounded": random.choice([True, True, True, False]),
        "confidence": round(random.uniform(0.01, 0.12), 4),
    }


# ---------------------------------------------------------------------------
# Real harness caller (uncomment / enable with --real flag)
# ---------------------------------------------------------------------------

async def _real_harness_call(harness, audio_bytes: bytes) -> dict:
    """Call the actual RAGHarness.run() and extract latency."""
    response = await harness.run(audio_bytes)
    return {
        "query": response.query_text,
        "answer": response.answer,
        "latency_breakdown": response.latency_breakdown.model_dump(),
        "strategy_used": response.strategy_used,
        "grounded": response.grounded,
        "confidence": response.confidence,
    }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

async def run_benchmark(n_queries: int = 150, use_real: bool = False):
    """Run n_queries through the pipeline and report percentile latencies."""

    queries = [SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)] for i in range(n_queries)]
    latencies = []
    stage_latencies = {
        "stt": [], "classify": [], "retrieve": [],
        "rerank": [], "guardrail": [], "generate": [],
    }
    errors = 0

    print(f"\n{'='*60}")
    print(f"  RECALL Benchmark -- {'REAL' if use_real else 'MOCK'} mode")
    print(f"  Queries: {n_queries}")
    print(f"{'='*60}\n")

    start_time = time.perf_counter()

    for i, query in enumerate(queries):
        try:
            if use_real:
                # TODO: Initialize real harness with loaded FAISS + BM25 stores
                # harness = RAGHarness(faiss_store, bm25_store)
                # result = await _real_harness_call(harness, audio_bytes)
                raise NotImplementedError(
                    "Real harness mode requires FAISS + BM25 stores to be loaded. "
                    "Run without --real for mock benchmarks."
                )
            else:
                result = await _mock_harness_call(query)

            lb = result["latency_breakdown"]
            latencies.append(lb["total_ms"])

            for stage in stage_latencies:
                key = f"{stage}_ms"
                if key in lb:
                    stage_latencies[stage].append(lb[key])

            # Progress indicator
            if (i + 1) % 25 == 0 or i == 0:
                print(f"  [{i+1:>{len(str(n_queries))}}/{n_queries}]  "
                      f"last={lb['total_ms']:.1f}ms  "
                      f"avg={np.mean(latencies):.1f}ms")

        except Exception as exc:
            errors += 1
            if errors <= 3:
                print(f"  [!] Query {i+1} failed: {exc}")

    wall_time = (time.perf_counter() - start_time) * 1000

    # --- Compute percentiles ---
    if not latencies:
        print("\n  [X] No successful queries. Cannot compute stats.")
        return {"error": "No successful queries", "n": n_queries, "errors": errors}

    p50, p70, p90, p95, p100 = np.percentile(latencies, [50, 70, 90, 95, 100])

    # Stage-level stats
    stage_stats = {}
    for stage, vals in stage_latencies.items():
        if vals:
            stage_stats[stage] = {
                "p50_ms": round(np.percentile(vals, 50), 2),
                "p70_ms": round(np.percentile(vals, 70), 2),
                "mean_ms": round(np.mean(vals), 2),
            }

    result = {
        "mode": "real" if use_real else "mock",
        "n": n_queries,
        "errors": errors,
        "success_rate": f"{((n_queries - errors) / n_queries * 100):.1f}%",
        "wall_time_ms": round(wall_time, 2),
        "total_latency": {
            "p50_ms": round(p50, 2),
            "p70_ms": round(p70, 2),
            "p90_ms": round(p90, 2),
            "p95_ms": round(p95, 2),
            "p100_ms": round(p100, 2),
            "mean_ms": round(np.mean(latencies), 2),
        },
        "stage_breakdown": stage_stats,
    }

    # --- Print report ---
    print(f"\n{'-'*60}")
    print(f"  Results")
    print(f"{'-'*60}")
    print(f"  Queries:       {n_queries} ({errors} errors)")
    print(f"  Wall time:     {wall_time:.0f}ms")
    print(f"")
    print(f"  Total Latency (end-to-end):")
    print(f"    P50:  {p50:>8.1f} ms")
    print(f"    P70:  {p70:>8.1f} ms")
    print(f"    P90:  {p90:>8.1f} ms")
    print(f"    P95:  {p95:>8.1f} ms")
    print(f"    P100: {p100:>8.1f} ms")
    print(f"    Mean: {np.mean(latencies):>8.1f} ms")
    print(f"")
    print(f"  Per-Stage P50:")
    for stage, stats in stage_stats.items():
        print(f"    {stage:<12s} {stats['p50_ms']:>8.1f} ms")
    print(f"{'-'*60}\n")

    # Save JSON report
    report_path = Path(__file__).resolve().parent / "benchmark_report.json"
    # Ensure JSON output uses UTF-8
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  Full report saved to: {report_path}\n")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RECALL latency benchmark")
    parser.add_argument("--n", type=int, default=150, help="Number of queries to run")
    parser.add_argument("--real", action="store_true", help="Use real harness (requires indices)")
    args = parser.parse_args()

    asyncio.run(run_benchmark(n_queries=args.n, use_real=args.real))