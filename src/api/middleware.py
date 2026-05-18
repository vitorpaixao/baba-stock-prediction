"""Structured-log + in-memory latency tracking middleware for FastAPI."""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections import Counter, deque
from threading import Lock
from typing import Deque

import numpy as np
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

log = logging.getLogger("api.access")


class MetricsStore:
    """Thread-safe in-process counters + latency ring buffer for /metrics."""

    def __init__(self, capacity: int = 1000) -> None:
        self._lock = Lock()
        self._latencies: Deque[float] = deque(maxlen=capacity)
        self._by_route: Counter[str] = Counter()
        self._total = 0

    def record(self, route: str, latency_ms: float) -> None:
        with self._lock:
            self._latencies.append(latency_ms)
            self._by_route[route] += 1
            self._total += 1

    def snapshot(self) -> dict:
        with self._lock:
            arr = np.asarray(self._latencies, dtype=np.float64) if self._latencies else np.array([])
            return {
                "total_requests": self._total,
                "by_route": dict(self._by_route),
                "latency_ms": {
                    "p50": float(np.percentile(arr, 50)) if arr.size else 0.0,
                    "p95": float(np.percentile(arr, 95)) if arr.size else 0.0,
                    "p99": float(np.percentile(arr, 99)) if arr.size else 0.0,
                    "mean": float(arr.mean()) if arr.size else 0.0,
                },
            }


METRICS = MetricsStore()


class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        t0 = time.perf_counter()
        status = 500
        try:
            response: Response = await call_next(request)
            status = response.status_code
            response.headers["x-request-id"] = request_id
            return response
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            route = request.url.path
            METRICS.record(route, elapsed_ms)
            log.info(json.dumps({
                "request_id": request_id,
                "method": request.method,
                "path": route,
                "status": status,
                "latency_ms": round(elapsed_ms, 2),
            }))
