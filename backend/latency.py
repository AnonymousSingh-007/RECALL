import time
from contextlib import contextmanager


class LatencyTracker:
    def __init__(self):
        self.stages: dict[str, float] = {}

    @contextmanager
    def track(self, name: str):
        start = time.perf_counter()
        yield
        self.stages[name] = (time.perf_counter() - start) * 1000

    @property
    def total_ms(self) -> float:
        return sum(self.stages.values())