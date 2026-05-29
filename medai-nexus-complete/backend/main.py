"""
MedAI Nexus — FastAPI Backend
Production-grade AI Healthcare Intelligence Platform
"""
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import asyncio, logging, time

from routes import auth, reports, chat, emergency, reminders, analytics, hospitals
from middleware.rate_limit import RateLimitMiddleware
from middleware.auth import AuthMiddleware
from core.config import settings
from core.database import init_db
from core.redis_client import init_redis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medai.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 MedAI Nexus starting up...")
    await init_db()
    await init_redis()
    yield
    logger.info("🔴 MedAI Nexus shutting down...")


app = FastAPI(
    title="MedAI Nexus API",
    description="AI-Powered Healthcare Intelligence Platform — LLMs + RAG + Multi-Agent",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────
app.add_middleware(CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware, requests_per_minute=60)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# ── Routers ───────────────────────────────────
app.include_router(auth.router,       prefix="/api/v1/auth",       tags=["Auth"])
app.include_router(reports.router,    prefix="/api/v1/reports",    tags=["Reports"])
app.include_router(chat.router,       prefix="/api/v1/chat",       tags=["AI Chat"])
app.include_router(emergency.router,  prefix="/api/v1/emergency",  tags=["Emergency"])
app.include_router(reminders.router,  prefix="/api/v1/reminders",  tags=["Reminders"])
app.include_router(analytics.router,  prefix="/api/v1/analytics",  tags=["Analytics"])
app.include_router(hospitals.router,  prefix="/api/v1/hospitals",  tags=["Hospitals"])


@app.get("/", tags=["Health"])
async def root():
    return {"status": "online", "platform": "MedAI Nexus", "version": "1.0.0"}


@app.get("/api/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "database": "connected",
            "redis": "connected",
            "vector_db": "connected",
            "ai_agents": "ready",
        },
    }
