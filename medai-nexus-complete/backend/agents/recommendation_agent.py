"""
MedAI Nexus — Recommendation Agent
Generates personalised doctor/specialist and lifestyle recommendations
based on the parsed medical report and abnormal values.
"""
from __future__ import annotations
import logging
from typing import Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger("medai.recommendation_agent")

SPECIALIST_MAP = {
    "hemoglobin": "Hematologist",
    "blood_sugar": "Endocrinologist / Diabetologist",
    "cholesterol": "Cardiologist",
    "blood_pressure": "Cardiologist",
    "creatinine": "Nephrologist",
    "tsh": "Endocrinologist",
    "vitamin_d": "General Physician",
}


class RecommendationAgent:
    SYSTEM = """You are a medical recommendation AI.
Given a parsed medical report with abnormal values and a health score,
generate clear, actionable recommendations.

Respond in JSON:
{
  "recommendations": ["rec1", "rec2", ...],
  "specialist_referrals": ["Hematologist", ...],
  "lifestyle_tips": ["tip1", ...],
  "follow_up_timeline": "2 weeks"
}

Be specific, empathetic, and practical. Max 6 recommendations."""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    async def generate(
        self,
        parsed_report: Dict,
        abnormal_values: List[Dict],
        health_score: int,
    ) -> Dict:
        # Fast path — no LLM, rule-based recommendations
        if not abnormal_values:
            return {
                "recommendations": [
                    "All values appear within normal range.",
                    "Continue regular health check-ups every 6 months.",
                    "Maintain a balanced diet and regular exercise.",
                ],
                "specialist_referrals": [],
                "lifestyle_tips": ["Stay hydrated", "Exercise 30 min/day"],
                "follow_up_timeline": "6 months",
            }

        # Rule-based specialist mapping
        specialists = list({
            SPECIALIST_MAP.get(v.get("name", "").lower().replace(" ", "_"), "General Physician")
            for v in abnormal_values
        })

        if not self.llm:
            return {
                "recommendations": [
                    f"Consult a {s} about your {v.get('name')} level ({v.get('value')} {v.get('unit', '')})"
                    for v, s in zip(abnormal_values[:3], specialists[:3])
                ] + ["Schedule a follow-up within 2 weeks."],
                "specialist_referrals": specialists[:3],
                "lifestyle_tips": [],
                "follow_up_timeline": "2 weeks" if health_score < 60 else "1 month",
            }

        try:
            import json, re
            prompt = (
                f"Health score: {health_score}/100\n"
                f"Abnormal values: {json.dumps(abnormal_values)}\n"
                "Generate recommendations."
            )
            resp = await self.llm.ainvoke([
                SystemMessage(content=self.SYSTEM),
                HumanMessage(content=prompt),
            ])
            raw = re.sub(r"```(?:json)?|```", "", resp.content).strip()
            result = json.loads(raw)
            return result
        except Exception as e:
            logger.error(f"[Recommendation Agent] {e}")
            return {
                "recommendations": ["Consult your doctor to review abnormal values."],
                "specialist_referrals": specialists[:2],
                "lifestyle_tips": [],
                "follow_up_timeline": "2 weeks",
            }
