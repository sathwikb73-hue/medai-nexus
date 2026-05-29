"""
MedAI Nexus — Emergency Detection Agent
Autonomous agent that triages symptom severity and
triggers emergency protocols when critical conditions detected.
"""
from __future__ import annotations
import re, logging
from typing import Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger("medai.emergency_agent")

# Fast keyword triage — runs before LLM for speed
CRITICAL_KEYWORDS = {
    "chest pain", "heart attack", "cardiac arrest", "can't breathe",
    "difficulty breathing", "stroke", "unconscious", "not breathing",
    "severe chest", "stroke symptoms", "face drooping", "arm weakness",
    "slurred speech", "overdose", "anaphylaxis", "severe allergic",
    "seizure", "convulsion", "choking", "cyanosis", "blue lips",
}
URGENT_KEYWORDS = {
    "high fever", "blood in urine", "coughing blood", "severe pain",
    "broken bone", "deep cut", "head injury", "fainting", "dizziness",
    "vomiting blood", "severe headache", "vision loss", "sudden blindness",
}


class EmergencyDetectionAgent:
    SYSTEM = """You are a medical emergency triage AI.

Assess the given text for emergency medical conditions.
Classify severity as: CRITICAL | URGENT | MODERATE | LOW

Respond in this exact JSON format:
{
  "is_emergency": true/false,
  "severity": "critical|urgent|moderate|low",
  "detected_conditions": ["condition1", "condition2"],
  "summary": "Brief assessment and immediate action steps",
  "call_ambulance": true/false,
  "immediate_actions": ["action1", "action2"]
}

CRITICAL = Life-threatening, call ambulance immediately (112/108)
URGENT   = Go to ER within 1 hour
MODERATE = See doctor today
LOW      = Monitor at home, see GP if worsens"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def fast_triage(self, text: str) -> bool:
        """Keyword-based fast triage — no LLM call needed."""
        lower = text.lower()
        return any(kw in lower for kw in CRITICAL_KEYWORDS)

    def is_urgent(self, text: str) -> bool:
        lower = text.lower()
        return any(kw in lower for kw in URGENT_KEYWORDS)

    async def assess(self, text: str) -> Dict:
        """Full LLM-powered emergency assessment."""
        # Fast path — skip LLM for obvious keywords
        if self.fast_triage(text):
            return {
                "is_emergency": True,
                "severity": "critical",
                "summary": "⚠️ CRITICAL symptoms detected. Call 112 (Emergency) or 108 (Ambulance) IMMEDIATELY. Do not wait.",
                "call_ambulance": True,
                "immediate_actions": [
                    "Call 112 or 108 immediately",
                    "Do not leave the person alone",
                    "Keep them calm and still",
                    "Do not give food or water",
                    "Unlock your door for paramedics",
                ],
            }

        if not self.llm:
            return {"is_emergency": False, "severity": "low", "summary": "No LLM available for detailed assessment."}

        try:
            import json
            response = await self.llm.ainvoke([
                SystemMessage(content=self.SYSTEM),
                HumanMessage(content=f"Assess these symptoms/text: {text[:1500]}"),
            ])
            raw = response.content.strip()
            # Strip markdown code fences if present
            raw = re.sub(r"```(?:json)?|```", "", raw).strip()
            result = json.loads(raw)
            logger.info(f"[Emergency Agent] Severity: {result.get('severity')} | Emergency: {result.get('is_emergency')}")
            return result
        except Exception as e:
            logger.error(f"[Emergency Agent] LLM assessment failed: {e}")
            # Conservative fallback
            urgent = self.is_urgent(text)
            return {
                "is_emergency": urgent,
                "severity": "urgent" if urgent else "low",
                "summary": "Unable to complete AI assessment. If symptoms are severe, call 112 immediately.",
                "call_ambulance": False,
            }


# ─────────────────────────────────────────────────────────
# Medical Analysis Agent
# ─────────────────────────────────────────────────────────
"""
MedAI Nexus — Medical Analysis Agent
Analyzes OCR-extracted text + RAG context to produce
structured medical report summaries with health scoring.
"""

NORMAL_RANGES = {
    "hemoglobin":      {"min": 12.0, "max": 17.5, "unit": "g/dL"},
    "wbc":             {"min": 4500,  "max": 11000, "unit": "/μL"},
    "blood_sugar":     {"min": 70,    "max": 100,   "unit": "mg/dL"},
    "cholesterol":     {"min": 0,     "max": 200,   "unit": "mg/dL"},
    "creatinine":      {"min": 0.6,   "max": 1.2,   "unit": "mg/dL"},
    "systolic_bp":     {"min": 90,    "max": 120,   "unit": "mmHg"},
    "diastolic_bp":    {"min": 60,    "max": 80,    "unit": "mmHg"},
    "vitamin_d":       {"min": 20,    "max": 100,   "unit": "ng/mL"},
    "tsh":             {"min": 0.4,   "max": 4.0,   "unit": "μIU/mL"},
    "ferritin":        {"min": 12,    "max": 300,   "unit": "ng/mL"},
}


class MedicalAnalysisAgent:
    SYSTEM = """You are an expert medical report analyst AI (MedAI).

Given OCR-extracted medical report text and relevant medical knowledge context,
produce a comprehensive analysis.

Respond in this exact JSON format:
{
  "report_type": "blood_test|mri|xray|prescription|other",
  "patient_info": {"name": "", "age": "", "date": ""},
  "key_findings": ["finding1", "finding2"],
  "abnormal_values": [
    {"name": "Hemoglobin", "value": "10.2", "unit": "g/dL", "status": "low",
     "normal_range": "12-17.5", "explanation": "May indicate anemia", "severity": "moderate"}
  ],
  "normal_values": [{"name": "", "value": "", "unit": ""}],
  "health_score": 72,
  "summary": "Plain-language summary for non-medical reader",
  "disease_indicators": ["possible anemia"],
  "recommended_actions": ["action1", "action2"],
  "follow_up_tests": ["test1"],
  "specialist_referral": "Hematologist"
}

health_score: 0-100 (100 = perfect health, <60 = needs attention)
Be empathetic. Explain medical terms in simple language."""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    async def analyze(self, ocr_text: str, rag_context: str = "") -> Dict:
        import json, re as _re
        prompt = f"""MEDICAL REPORT TEXT (OCR Extracted):
{ocr_text[:3000]}

RELEVANT MEDICAL KNOWLEDGE (RAG Retrieved):
{rag_context[:1500]}

Analyze the report and respond in the required JSON format."""

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=self.SYSTEM),
                HumanMessage(content=prompt),
            ])
            raw = _re.sub(r"```(?:json)?|```", "", response.content).strip()
            result = json.loads(raw)
            # Ensure health_score is int
            result["health_score"] = int(result.get("health_score", 75))
            result["parsed"]   = {v["name"]: v["value"] for v in result.get("abnormal_values", [])}
            result["abnormal"] = result.get("abnormal_values", [])
            logger.info(f"[Medical Agent] Score={result['health_score']} | Abnormals={len(result['abnormal'])}")
            return result
        except Exception as e:
            logger.error(f"[Medical Agent] Analysis failed: {e}")
            return {
                "health_score": 70, "summary": "Analysis could not be completed. Please consult your doctor.",
                "abnormal_values": [], "parsed": {}, "abnormal": [],
                "recommended_actions": ["Consult your healthcare provider for interpretation."],
            }
