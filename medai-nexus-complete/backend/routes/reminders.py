"""
MedAI Nexus — Reminders Routes
POST / · GET / · PATCH /{id}/taken · DELETE /{id}
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime
from typing import Optional
import uuid

from core.database import get_db
from middleware.auth import get_current_user
from models.models import User, Reminder, Medicine, ReminderFreq

router = APIRouter()


class ReminderIn(BaseModel):
    medicine_name: str
    dosage: str | None = None
    remind_at: datetime
    frequency: str = "daily"
    channel: str = "email"


@router.post("/", status_code=201)
async def create_reminder(
    payload: ReminderIn,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Create medicine entry
    med = Medicine(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        name=payload.medicine_name,
        dosage=payload.dosage,
    )
    db.add(med)
    await db.flush()

    reminder = Reminder(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        medicine_id=med.id,
        remind_at=payload.remind_at,
        frequency=ReminderFreq(payload.frequency),
        channel=payload.channel,
    )
    db.add(reminder)
    await db.commit()
    return {"reminder_id": reminder.id, "medicine": payload.medicine_name, "remind_at": payload.remind_at.isoformat()}


@router.get("/")
async def list_reminders(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Reminder)
        .where(Reminder.user_id == current_user.id)
        .order_by(Reminder.remind_at.asc())
        .limit(50)
    )
    reminders = result.scalars().all()
    return [
        {"id": r.id, "remind_at": r.remind_at.isoformat(),
         "frequency": r.frequency, "is_taken": r.is_taken, "channel": r.channel}
        for r in reminders
    ]


@router.patch("/{reminder_id}/taken")
async def mark_taken(
    reminder_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Reminder).where(Reminder.id == reminder_id, Reminder.user_id == current_user.id)
    )
    reminder = result.scalar_one_or_none()
    if not reminder:
        raise HTTPException(404, "Reminder not found")
    reminder.is_taken = True
    await db.commit()
    return {"status": "marked as taken"}


@router.delete("/{reminder_id}", status_code=204)
async def delete_reminder(
    reminder_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Reminder).where(Reminder.id == reminder_id, Reminder.user_id == current_user.id)
    )
    reminder = result.scalar_one_or_none()
    if not reminder:
        raise HTTPException(404, "Reminder not found")
    await db.delete(reminder)
    await db.commit()
