"""
MedAI Nexus — Analytics Routes
GET /dashboard · GET /trends · GET /risks · GET /score
"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import datetime, timedelta

from core.database import get_db
from middleware.auth import get_current_user
from models.models import User, MedicalReport, EmergencyLog, AIInsight

router = APIRouter()


@router.get("/dashboard")
async def dashboard(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Latest report health score
    r = await db.execute(
        select(MedicalReport)
        .where(MedicalReport.user_id == current_user.id,
               MedicalReport.health_score.isnot(None))
        .order_by(MedicalReport.created_at.desc())
        .limit(1)
    )
    latest = r.scalar_one_or_none()

    # Report count
    count_r = await db.execute(
        select(func.count(MedicalReport.id))
        .where(MedicalReport.user_id == current_user.id)
    )
    report_count = count_r.scalar()

    # Emergency count
    emrg_r = await db.execute(
        select(func.count(EmergencyLog.id))
        .where(EmergencyLog.user_id == current_user.id)
    )
    emrg_count = emrg_r.scalar()

    return {
        "health_score":    latest.health_score if latest else None,
        "total_reports":   report_count,
        "emergency_events": emrg_count,
        "last_report_date": latest.created_at.isoformat() if latest else None,
        "vitals": latest.parsed_data if latest else {},
        "insights": latest.ai_insights if latest else {},
    }


@router.get("/trends")
async def trends(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    days: int = 180,
):
    since = datetime.utcnow() - timedelta(days=days)
    r = await db.execute(
        select(MedicalReport)
        .where(MedicalReport.user_id == current_user.id,
               MedicalReport.created_at >= since,
               MedicalReport.health_score.isnot(None))
        .order_by(MedicalReport.created_at.asc())
    )
    reports = r.scalars().all()
    return [
        {
            "date":  rp.created_at.strftime("%b %Y"),
            "score": rp.health_score,
            "data":  rp.parsed_data or {},
        }
        for rp in reports
    ]


@router.get("/score")
async def health_score(current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    r = await db.execute(
        select(MedicalReport)
        .where(MedicalReport.user_id == current_user.id, MedicalReport.health_score.isnot(None))
        .order_by(MedicalReport.created_at.desc()).limit(1)
    )
    latest = r.scalar_one_or_none()
    return {"score": latest.health_score if latest else None, "updated_at": latest.created_at.isoformat() if latest else None}


@router.get("/risks")
async def risk_predictions(current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    r = await db.execute(
        select(MedicalReport)
        .where(MedicalReport.user_id == current_user.id, MedicalReport.ai_insights.isnot(None))
        .order_by(MedicalReport.created_at.desc()).limit(1)
    )
    latest = r.scalar_one_or_none()
    if not latest or not latest.ai_insights:
        return {"risks": []}
    return {"risks": latest.ai_insights.get("abnormal_values", [])}
