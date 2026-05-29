"""
MedAI Nexus — Emergency Routes
POST /assess · GET /logs · POST /alert
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime
from typing import List
import uuid

from core.database import get_db
from middleware.auth import get_current_user
from models.models import User, EmergencyLog

router = APIRouter()


class SymptomPayload(BaseModel):
    symptoms: List[str]
    latitude: float | None = None
    longitude: float | None = None


@router.post("/assess")
async def assess_emergency(
    payload: SymptomPayload,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from agents.emergency_agent import EmergencyDetectionAgent
    agent = EmergencyDetectionAgent(None)  # uses fast triage

    text = ", ".join(payload.symptoms)
    result = await agent.assess(text)

    # Log if significant
    if result.get("severity") in ("critical", "urgent"):
        log = EmergencyLog(
            id=str(uuid.uuid4()),
            user_id=current_user.id,
            symptoms=payload.symptoms,
            severity=result.get("severity", "low"),
            ai_assessment=result.get("summary", ""),
            location_lat=payload.latitude,
            location_lon=payload.longitude,
        )
        db.add(log)
        await db.commit()

    return result


@router.get("/logs")
async def emergency_logs(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(EmergencyLog)
        .where(EmergencyLog.user_id == current_user.id)
        .order_by(EmergencyLog.created_at.desc())
        .limit(20)
    )
    logs = result.scalars().all()
    return [
        {
            "id": l.id, "symptoms": l.symptoms, "severity": l.severity,
            "ai_assessment": l.ai_assessment, "created_at": l.created_at.isoformat(),
        }
        for l in logs
    ]
