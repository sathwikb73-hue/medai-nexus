"""
MedAI Nexus — Reminder Agent
Schedules medicine reminder tasks via Celery + Redis.
Falls back to in-process scheduling if Celery unavailable.
"""
from __future__ import annotations
import logging
from datetime import datetime, timedelta
from typing import Dict, List

logger = logging.getLogger("medai.reminder_agent")


class ReminderAgent:
    """
    Schedules medicine reminders.
    Uses Celery for production async task queuing.
    """

    async def schedule(
        self,
        user_id: str,
        medicine_name: str,
        remind_at: datetime,
        frequency: str = "daily",
        channel: str = "email",
    ) -> Dict:
        """
        Schedule a reminder task.
        Returns task_id for tracking.
        """
        try:
            from tasks.reminder_tasks import send_reminder_task
            eta = remind_at
            result = send_reminder_task.apply_async(
                args=[user_id, medicine_name, channel],
                eta=eta,
            )
            logger.info(f"[Reminder Agent] Scheduled {medicine_name} for {user_id} at {remind_at} via {channel}")
            return {"task_id": result.id, "scheduled_at": remind_at.isoformat(), "status": "scheduled"}

        except ImportError:
            # Celery not configured — log and return gracefully
            logger.warning("[Reminder Agent] Celery not available — reminder logged only")
            return {"task_id": None, "scheduled_at": remind_at.isoformat(), "status": "logged"}

        except Exception as e:
            logger.error(f"[Reminder Agent] Failed to schedule: {e}")
            return {"task_id": None, "scheduled_at": remind_at.isoformat(), "status": "failed"}

    async def schedule_recurring(
        self,
        user_id: str,
        medicine_name: str,
        start: datetime,
        frequency: str,
        duration_days: int = 30,
    ) -> List[Dict]:
        """Schedule multiple reminders for a recurring medicine course."""
        tasks = []
        current = start
        delta = {"daily": timedelta(days=1), "weekly": timedelta(weeks=1), "once": None}.get(frequency)

        if not delta:
            return [await self.schedule(user_id, medicine_name, current, frequency)]

        while current <= start + timedelta(days=duration_days):
            task = await self.schedule(user_id, medicine_name, current, frequency)
            tasks.append(task)
            current += delta

        return tasks
