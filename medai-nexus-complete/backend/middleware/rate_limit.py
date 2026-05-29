"""
MedAI Nexus — Rate Limit Middleware
Redis-backed sliding window rate limiter.
Default: 60 requests/minute per IP.
"""
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
import time, logging

logger = logging.getLogger("medai.ratelimit")

EXEMPT_PATHS = {"/", "/api/health", "/api/docs", "/api/redoc", "/openapi.json"}


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.rpm = requests_per_minute
        self.window = 60

    async def dispatch(self, request: Request, call_next):
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        ip = request.client.host if request.client else "unknown"
        key = f"rl:{ip}"

        try:
            from core.redis_client import redis_client
            if redis_client:
                now = int(time.time())
                pipe = redis_client.pipeline()
                pipe.zadd(key, {str(now): now})
                pipe.zremrangebyscore(key, 0, now - self.window)
                pipe.zcard(key)
                pipe.expire(key, self.window)
                results = await pipe.execute()
                count = results[2]
                if count > self.rpm:
                    logger.warning(f"[RateLimit] {ip} exceeded {self.rpm} rpm")
                    return JSONResponse(
                        status_code=429,
                        content={"detail": "Too many requests. Please slow down."},
                        headers={"Retry-After": "60", "X-RateLimit-Limit": str(self.rpm)},
                    )
        except Exception:
            pass  # Redis unavailable — fail open

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.rpm)
        return response
