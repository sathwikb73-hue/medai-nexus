"""
MedAI Nexus — Redis Async Client
Connection pool for caching, rate limiting, and Celery broker.
"""
import redis.asyncio as aioredis
from core.config import settings
import logging

logger = logging.getLogger("medai.redis")

redis_client: aioredis.Redis | None = None


async def init_redis():
    global redis_client
    redis_client = aioredis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
        max_connections=20,
    )
    await redis_client.ping()
    logger.info("✅ Redis connected")


async def get_redis() -> aioredis.Redis:
    """FastAPI dependency."""
    if redis_client is None:
        raise RuntimeError("Redis not initialised")
    return redis_client


async def cache_set(key: str, value: str, ttl: int = 300):
    if redis_client:
        await redis_client.setex(key, ttl, value)


async def cache_get(key: str) -> str | None:
    if redis_client:
        return await redis_client.get(key)
    return None


async def cache_delete(key: str):
    if redis_client:
        await redis_client.delete(key)
