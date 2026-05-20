"""Structured-log middleware for FastAPI.

HTTP-level counters and latency tracking now live in Prometheus
(see ``prometheus_fastapi_instrumentator`` wiring in ``src/api/main.py``).
This middleware only emits one JSON access-log line per request.
"""
from __future__ import annotations

import json
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

log = logging.getLogger("api.access")


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
            log.info(json.dumps({
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": status,
                "latency_ms": round(elapsed_ms, 2),
            }))
