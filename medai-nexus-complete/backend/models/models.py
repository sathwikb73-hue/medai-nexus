"""
MedAI Nexus — Database Models
Complete schema: users, reports, conversations, medicines,
reminders, emergency_logs, ai_insights, hospitals
"""
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime,
    Text, ForeignKey, JSON, Enum as SAEnum, Index
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid, enum

Base = declarative_base()


def gen_uuid():
    return str(uuid.uuid4())


# ─── Enums ────────────────────────────────────
class ReportStatus(str, enum.Enum):
    pending   = "pending"
    ocr       = "ocr"
    embedding = "embedding"
    analyzed  = "analyzed"
    failed    = "failed"

class Severity(str, enum.Enum):
    low      = "low"
    moderate = "moderate"
    high     = "high"
    critical = "critical"

class ReminderFreq(str, enum.Enum):
    once    = "once"
    daily   = "daily"
    weekly  = "weekly"
    custom  = "custom"


# ─── Users ────────────────────────────────────
class User(Base):
    __tablename__ = "users"
    id            = Column(String, primary_key=True, default=gen_uuid)
    email         = Column(String(255), unique=True, nullable=False, index=True)
    full_name     = Column(String(255))
    hashed_pw     = Column(String(512))
    oauth_provider= Column(String(32))          # google / github
    oauth_id      = Column(String(255))
    is_active     = Column(Boolean, default=True)
    is_premium    = Column(Boolean, default=False)
    role          = Column(String(32), default="patient")  # patient | admin
    dob           = Column(DateTime)
    gender        = Column(String(16))
    blood_group   = Column(String(8))
    created_at    = Column(DateTime, default=datetime.utcnow)
    updated_at    = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    reports        = relationship("MedicalReport",   back_populates="user", cascade="all, delete")
    conversations  = relationship("Conversation",    back_populates="user", cascade="all, delete")
    reminders      = relationship("Reminder",        back_populates="user", cascade="all, delete")
    insights       = relationship("AIInsight",       back_populates="user", cascade="all, delete")
    emergency_logs = relationship("EmergencyLog",    back_populates="user", cascade="all, delete")

    __table_args__ = (Index("ix_users_email_active", "email", "is_active"),)


# ─── Medical Reports ──────────────────────────
class MedicalReport(Base):
    __tablename__ = "reports"
    id            = Column(String, primary_key=True, default=gen_uuid)
    user_id       = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    filename      = Column(String(512))
    file_path     = Column(String(1024))
    file_type     = Column(String(16))           # pdf | image
    report_type   = Column(String(64))           # blood_test | mri | prescription | xray
    status        = Column(SAEnum(ReportStatus), default=ReportStatus.pending)
    ocr_text      = Column(Text)
    ocr_confidence= Column(Float)
    parsed_data   = Column(JSON)                 # structured extracted values
    embedding_ids = Column(JSON)                 # list of chroma/pinecone chunk IDs
    ai_summary    = Column(Text)
    ai_insights   = Column(JSON)                 # abnormal values, recommendations
    health_score  = Column(Integer)
    created_at    = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="reports")

    __table_args__ = (
        Index("ix_reports_user_created", "user_id", "created_at"),
        Index("ix_reports_status", "status"),
    )


# ─── Conversations ────────────────────────────
class Conversation(Base):
    __tablename__ = "conversations"
    id         = Column(String, primary_key=True, default=gen_uuid)
    user_id    = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title      = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user     = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete",
                            order_by="Message.created_at")


class Message(Base):
    __tablename__ = "messages"
    id              = Column(String, primary_key=True, default=gen_uuid)
    conversation_id = Column(String, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    role            = Column(String(16))         # user | assistant | system
    content         = Column(Text, nullable=False)
    tokens_used     = Column(Integer, default=0)
    rag_context     = Column(JSON)               # which chunks were retrieved
    is_emergency    = Column(Boolean, default=False)
    created_at      = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")


# ─── Medicines & Reminders ────────────────────
class Medicine(Base):
    __tablename__ = "medicines"
    id           = Column(String, primary_key=True, default=gen_uuid)
    user_id      = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name         = Column(String(255), nullable=False)
    dosage       = Column(String(128))
    purpose      = Column(String(512))
    side_effects = Column(Text)
    instructions = Column(Text)
    start_date   = Column(DateTime)
    end_date     = Column(DateTime)
    is_active    = Column(Boolean, default=True)
    created_at   = Column(DateTime, default=datetime.utcnow)

    reminders = relationship("Reminder", back_populates="medicine", cascade="all, delete")


class Reminder(Base):
    __tablename__ = "reminders"
    id            = Column(String, primary_key=True, default=gen_uuid)
    user_id       = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    medicine_id   = Column(String, ForeignKey("medicines.id", ondelete="CASCADE"))
    remind_at     = Column(DateTime, nullable=False, index=True)
    frequency     = Column(SAEnum(ReminderFreq), default=ReminderFreq.daily)
    channel       = Column(String(16), default="email")   # email | sms | push
    is_sent       = Column(Boolean, default=False)
    is_taken      = Column(Boolean, default=False)
    created_at    = Column(DateTime, default=datetime.utcnow)

    user     = relationship("User",     back_populates="reminders")
    medicine = relationship("Medicine", back_populates="reminders")


# ─── Emergency Logs ───────────────────────────
class EmergencyLog(Base):
    __tablename__ = "emergency_logs"
    id            = Column(String, primary_key=True, default=gen_uuid)
    user_id       = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    symptoms      = Column(JSON)
    severity      = Column(SAEnum(Severity))
    ai_assessment = Column(Text)
    location_lat  = Column(Float)
    location_lon  = Column(Float)
    hospitals_alerted = Column(JSON)
    resolved_at   = Column(DateTime)
    created_at    = Column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="emergency_logs")


# ─── AI Insights ──────────────────────────────
class AIInsight(Base):
    __tablename__ = "ai_insights"
    id           = Column(String, primary_key=True, default=gen_uuid)
    user_id      = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    insight_type = Column(String(64))            # risk_prediction | trend | recommendation
    title        = Column(String(255))
    content      = Column(Text)
    data         = Column(JSON)
    score        = Column(Float)
    is_read      = Column(Boolean, default=False)
    created_at   = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="insights")


# ─── Hospital Cache ───────────────────────────
class Hospital(Base):
    __tablename__ = "hospitals"
    id           = Column(String, primary_key=True, default=gen_uuid)
    place_id     = Column(String(255), unique=True)
    name         = Column(String(512))
    address      = Column(Text)
    latitude     = Column(Float)
    longitude    = Column(Float)
    phone        = Column(String(32))
    rating       = Column(Float)
    specialties  = Column(JSON)
    is_emergency = Column(Boolean, default=False)
    updated_at   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
