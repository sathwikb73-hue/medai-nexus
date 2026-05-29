"""
MedAI Nexus — Core Configuration
All environment variables with typed defaults
"""
from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    # App
    APP_NAME: str = "MedAI Nexus"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    SECRET_KEY: str = os.getenv("SECRET_KEY", "change-me-in-production-32-chars-min")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hrs

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://medai:medai@localhost:5432/medai_nexus")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # AI / LLM
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    LLM_MODEL: str = "gpt-4o"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    MAX_TOKENS: int = 2048
    TEMPERATURE: float = 0.3

    # Vector DB (Chroma / Pinecone)
    VECTOR_DB: str = "chroma"                        # "chroma" | "pinecone"
    CHROMA_HOST: str = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT: int = 8000
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX: str = "medai-embeddings"

    # Storage
    UPLOAD_DIR: str = "/tmp/medai_uploads"
    MAX_FILE_SIZE_MB: int = 20
    ALLOWED_EXTENSIONS: List[str] = ["pdf", "png", "jpg", "jpeg", "webp"]

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "https://medai-nexus.vercel.app",
    ]

    # Maps
    GOOGLE_MAPS_KEY: str = os.getenv("GOOGLE_MAPS_KEY", "")

    # Email (Reminders)
    SMTP_HOST: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT: int = 587
    SMTP_USER: str = os.getenv("SMTP_USER", "")
    SMTP_PASS: str = os.getenv("SMTP_PASS", "")

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
