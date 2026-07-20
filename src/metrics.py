"""简单内存指标收集.

用于记录请求数、延迟分布、LLM 调用情况等。
"""

import time
from collections import deque
from threading import Lock
from typing import Deque, Dict


class MetricsCollector:
    """指标收集器."""

    def __init__(self, max_latency_samples: int = 10000):
        self.max_latency_samples = max_latency_samples
        self._lock = Lock()
        self._request_count = 0
        self._error_count = 0
        self._llm_call_count = 0
        self._llm_error_count = 0
        self._latencies_ms: Deque[float] = deque(maxlen=max_latency_samples)
        self._llm_latencies_ms: Deque[float] = deque(maxlen=max_latency_samples)

    def record_request(self, latency_ms: float, is_error: bool = False) -> None:
        with self._lock:
            self._request_count += 1
            self._latencies_ms.append(latency_ms)
            if is_error:
                self._error_count += 1

    def record_llm_call(self, latency_ms: float, is_error: bool = False) -> None:
        with self._lock:
            self._llm_call_count += 1
            self._llm_latencies_ms.append(latency_ms)
            if is_error:
                self._llm_error_count += 1

    def _percentile(self, values: Deque[float], p: float) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * p
        f = int(k)
        c = min(f + 1, len(sorted_values) - 1)
        return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)

    def snapshot(self) -> Dict:
        with self._lock:
            return {
                "request_count": self._request_count,
                "error_count": self._error_count,
                "error_rate": round(self._error_count / max(self._request_count, 1), 4),
                "llm_call_count": self._llm_call_count,
                "llm_error_count": self._llm_error_count,
                "llm_error_rate": round(self._llm_error_count / max(self._llm_call_count, 1), 4),
                "latency_ms": {
                    "p50": self._percentile(self._latencies_ms, 0.5),
                    "p95": self._percentile(self._latencies_ms, 0.95),
                    "p99": self._percentile(self._latencies_ms, 0.99),
                    "avg": sum(self._latencies_ms) / max(len(self._latencies_ms), 1),
                },
                "llm_latency_ms": {
                    "p50": self._percentile(self._llm_latencies_ms, 0.5),
                    "p95": self._percentile(self._llm_latencies_ms, 0.95),
                    "p99": self._percentile(self._llm_latencies_ms, 0.99),
                    "avg": sum(self._llm_latencies_ms) / max(len(self._llm_latencies_ms), 1),
                },
            }


_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    return _metrics
