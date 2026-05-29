"""
MedAI Nexus — Medical Analysis Agent
Analyses OCR-extracted report text + RAG context
to produce structured medical insights with health scoring.
(Full implementation is in emergency_agent.py — this re-exports it cleanly.)
"""
from agents.emergency_agent import MedicalAnalysisAgent

__all__ = ["MedicalAnalysisAgent"]
