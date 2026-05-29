"""
MedAI Nexus — Reports Routes
POST /upload · GET / · GET /{id} · DELETE /{id}
Full OCR + RAG + LLM pipeline on upload.
"""
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pathlib import Path
from datetime import datetime
import uuid, shutil, logging, os

from core.database import get_db
from core.config import settings
from middleware.auth import get_current_user
from models.models import User, MedicalReport, ReportStatus

router = APIRouter()
logger = logging.getLogger("medai.reports")

UPLOAD_DIR = Path(settings.UPLOAD_DIR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
MAX_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024


@router.post("/upload")
async def upload_report(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Validate
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(422, f"File type {ext} not supported. Use: {ALLOWED}")

    content = await file.read()
    if len(content) > MAX_BYTES:
        raise HTTPException(413, f"File exceeds {settings.MAX_FILE_SIZE_MB} MB limit")

    # Save file
    report_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{report_id}{ext}"
    with open(file_path, "wb") as f:
        f.write(content)

    # Create DB record
    report = MedicalReport(
        id=report_id,
        user_id=current_user.id,
        filename=file.filename,
        file_path=str(file_path),
        file_type="pdf" if ext == ".pdf" else "image",
        status=ReportStatus.pending,
    )
    db.add(report)
    await db.commit()
    await db.refresh(report)

    # Run analysis in background
    background_tasks.add_task(_analyze_report, report_id, str(file_path), current_user.id)

    return {
        "report_id": report_id,
        "filename": file.filename,
        "status": "pending",
        "message": "Report uploaded. AI analysis started in background.",
    }


@router.get("/")
async def list_reports(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    skip: int = 0,
    limit: int = 20,
):
    result = await db.execute(
        select(MedicalReport)
        .where(MedicalReport.user_id == current_user.id)
        .order_by(MedicalReport.created_at.desc())
        .offset(skip).limit(limit)
    )
    reports = result.scalars().all()
    return [
        {
            "id": r.id, "filename": r.filename, "report_type": r.report_type,
            "status": r.status, "health_score": r.health_score,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in reports
    ]


@router.get("/{report_id}")
async def get_report(
    report_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(MedicalReport).where(
            MedicalReport.id == report_id,
            MedicalReport.user_id == current_user.id,
        )
    )
    report = result.scalar_one_or_none()
    if not report:
        raise HTTPException(404, "Report not found")

    return {
        "id": report.id,
        "filename": report.filename,
        "status": report.status,
        "report_type": report.report_type,
        "ocr_confidence": report.ocr_confidence,
        "ai_summary": report.ai_summary,
        "ai_insights": report.ai_insights,
        "health_score": report.health_score,
        "parsed_data": report.parsed_data,
        "created_at": report.created_at.isoformat() if report.created_at else None,
    }


@router.delete("/{report_id}", status_code=204)
async def delete_report(
    report_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(MedicalReport).where(
            MedicalReport.id == report_id,
            MedicalReport.user_id == current_user.id,
        )
    )
    report = result.scalar_one_or_none()
    if not report:
        raise HTTPException(404, "Report not found")

    # Remove file from disk
    try:
        if report.file_path and os.path.exists(report.file_path):
            os.remove(report.file_path)
    except Exception as e:
        logger.warning(f"Could not delete file {report.file_path}: {e}")

    await db.delete(report)
    await db.commit()


async def _analyze_report(report_id: str, file_path: str, user_id: str):
    """Background task: run full OCR → RAG → LLM pipeline."""
    from core.database import AsyncSessionLocal
    from agents.orchestrator import orchestrator

    async with AsyncSessionLocal() as db:
        try:
            result = await db.execute(select(MedicalReport).where(MedicalReport.id == report_id))
            report = result.scalar_one_or_none()
            if not report:
                return

            report.status = ReportStatus.ocr
            await db.commit()

            analysis = await orchestrator.run_report_analysis(user_id, file_path)

            report.status       = ReportStatus.analyzed
            report.ocr_text     = analysis.get("ocr_text", "")
            report.ai_summary   = analysis.get("summary", "")
            report.ai_insights  = {
                "abnormal_values":  analysis.get("abnormal_values", []),
                "recommendations":  analysis.get("recommendations", []),
                "is_emergency":     analysis.get("is_emergency", False),
            }
            report.health_score = analysis.get("health_score", 0)
            report.parsed_data  = analysis.get("parsed_report", {})
            await db.commit()
            logger.info(f"[Reports] Analysis complete for {report_id}")

        except Exception as e:
            logger.error(f"[Reports] Analysis failed for {report_id}: {e}")
            result = await db.execute(select(MedicalReport).where(MedicalReport.id == report_id))
            report = result.scalar_one_or_none()
            if report:
                report.status = ReportStatus.failed
                await db.commit()
