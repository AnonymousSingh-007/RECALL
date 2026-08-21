"""Run N queries through the full pipeline, report P50/P70/P100."""
import json
import numpy as np

def run_benchmark(n_queries: int = 150):
    latencies = []
    # TODO Day 4: loop over sample queries, call harness.run(), record total_ms
    p50, p70, p100 = np.percentile(latencies, [50, 70, 100]) if latencies else (0, 0, 0)
    result = {"p50_ms": p50, "p70_ms": p70, "p100_ms": p100, "n": n_queries}
    print(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    run_benchmark()