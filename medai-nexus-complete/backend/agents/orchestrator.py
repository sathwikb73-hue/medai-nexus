"""
MedAI Nexus — Multi-Agent AI Orchestration
LangGraph workflow connecting all autonomous agents:
  OCR → Medical Analysis → RAG Retrieval → Emergency Detection → Recommendation
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import asyncio, logging

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from agents.ocr_agent import OCRAgent
from agents.medical_agent import MedicalAnalysisAgent
from agents.rag_agent import RAGRetrievalAgent
from agents.emergency_agent import EmergencyDetectionAgent
from agents.recommendation_agent import RecommendationAgent
from agents.reminder_agent import ReminderAgent
from core.config import settings

logger = logging.getLogger("medai.orchestrator")


# ─── Shared Agent State ───────────────────────
@dataclass
class MedAIState:
    # Inputs
    user_id: str = ""
    input_type: str = ""              # "report" | "chat" | "symptom"
    raw_file_path: Optional[str] = None
    user_query: Optional[str] = None
    conversation_history: List[Dict] = field(default_factory=list)

    # OCR stage
    ocr_text: str = ""
    ocr_confidence: float = 0.0

    # RAG stage
    retrieved_chunks: List[str] = field(default_factory=list)
    retrieved_sources: List[str] = field(default_factory=list)

    # Analysis stage
    parsed_report: Dict[str, Any] = field(default_factory=dict)
    abnormal_values: List[Dict] = field(default_factory=list)

    # Emergency stage
    is_emergency: bool = False
    severity: str = "low"
    emergency_summary: str = ""

    # Final outputs
    ai_response: str = ""
    recommendations: List[str] = field(default_factory=list)
    health_score: int = 0
    error: Optional[str] = None


# ─── Agent Nodes ──────────────────────────────
class AgentOrchestrator:
    """
    LangGraph-based orchestrator connecting all MedAI agents.
    Supports two primary workflows:
      1. report_flow: file → OCR → RAG → MedicalAnalysis → Recommendation
      2. chat_flow:   query → EmergencyCheck → RAG → ChatResponse
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=settings.TEMPERATURE,
            streaming=True,
        )
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY,
        )
        self.ocr         = OCRAgent()
        self.medical     = MedicalAnalysisAgent(self.llm)
        self.rag         = RAGRetrievalAgent(self.embeddings)
        self.emergency   = EmergencyDetectionAgent(self.llm)
        self.recommender = RecommendationAgent(self.llm)
        self.reminder    = ReminderAgent()

        self._report_graph = self._build_report_graph()
        self._chat_graph   = self._build_chat_graph()

    # ── Report Workflow ────────────────────────
    def _build_report_graph(self) -> StateGraph:
        g = StateGraph(MedAIState)

        g.add_node("ocr",            self._node_ocr)
        g.add_node("rag_retrieval",  self._node_rag)
        g.add_node("medical_parse",  self._node_medical)
        g.add_node("emergency_check",self._node_emergency)
        g.add_node("recommend",      self._node_recommend)
        g.add_node("finalize",       self._node_finalize)

        g.set_entry_point("ocr")
        g.add_edge("ocr",             "rag_retrieval")
        g.add_edge("rag_retrieval",   "medical_parse")
        g.add_edge("medical_parse",   "emergency_check")
        g.add_conditional_edges("emergency_check", self._route_emergency,
            {"emergency": "recommend", "normal": "recommend"})
        g.add_edge("recommend", "finalize")
        g.add_edge("finalize",  END)
        return g.compile()

    # ── Chat Workflow ──────────────────────────
    def _build_chat_graph(self) -> StateGraph:
        g = StateGraph(MedAIState)

        g.add_node("emergency_triage", self._node_emergency)
        g.add_node("rag_retrieval",    self._node_rag_chat)
        g.add_node("chat_response",    self._node_chat)
        g.add_node("finalize",         self._node_finalize)

        g.set_entry_point("emergency_triage")
        g.add_conditional_edges("emergency_triage", self._route_emergency,
            {"emergency": "chat_response", "normal": "rag_retrieval"})
        g.add_edge("rag_retrieval", "chat_response")
        g.add_edge("chat_response", "finalize")
        g.add_edge("finalize",      END)
        return g.compile()

    # ── Node Implementations ───────────────────
    async def _node_ocr(self, state: MedAIState) -> MedAIState:
        logger.info(f"[OCR Agent] Processing {state.raw_file_path}")
        result = await self.ocr.extract(state.raw_file_path)
        state.ocr_text       = result["text"]
        state.ocr_confidence = result["confidence"]
        return state

    async def _node_rag(self, state: MedAIState) -> MedAIState:
        logger.info("[RAG Agent] Retrieving medical context for report")
        result = await self.rag.retrieve(
            query=state.ocr_text[:1000],
            top_k=5,
            filter_metadata={"type": "medical_reference"},
        )
        state.retrieved_chunks  = result["chunks"]
        state.retrieved_sources = result["sources"]
        return state

    async def _node_rag_chat(self, state: MedAIState) -> MedAIState:
        logger.info("[RAG Agent] Retrieving medical context for chat")
        result = await self.rag.retrieve(
            query=state.user_query,
            top_k=4,
        )
        state.retrieved_chunks  = result["chunks"]
        state.retrieved_sources = result["sources"]
        return state

    async def _node_medical(self, state: MedAIState) -> MedAIState:
        logger.info("[Medical Agent] Analyzing report")
        result = await self.medical.analyze(
            ocr_text=state.ocr_text,
            rag_context="\n---\n".join(state.retrieved_chunks),
        )
        state.parsed_report   = result["parsed"]
        state.abnormal_values = result["abnormal"]
        state.health_score    = result["health_score"]
        state.ai_response     = result["summary"]
        return state

    async def _node_emergency(self, state: MedAIState) -> MedAIState:
        text = state.user_query or state.ocr_text
        result = await self.emergency.assess(text)
        state.is_emergency       = result["is_emergency"]
        state.severity           = result["severity"]
        state.emergency_summary  = result["summary"]
        if state.is_emergency:
            state.ai_response = f"🚨 EMERGENCY DETECTED\n\n{result['summary']}\n\nCall 112 immediately."
        return state

    async def _node_chat(self, state: MedAIState) -> MedAIState:
        logger.info("[LLM] Generating chat response")
        context = "\n---\n".join(state.retrieved_chunks)
        history = [
            HumanMessage(content=m["content"]) if m["role"] == "user"
            else SystemMessage(content=m["content"])
            for m in state.conversation_history[-8:]
        ]
        system = SystemMessage(content=(
            "You are MedAI, an expert AI healthcare assistant. Use the medical reference context below "
            "to answer accurately. Be empathetic, clear, and structured.\n\n"
            f"MEDICAL CONTEXT:\n{context}"
        ))
        response = await self.llm.ainvoke([system, *history, HumanMessage(content=state.user_query)])
        state.ai_response = response.content
        return state

    async def _node_recommend(self, state: MedAIState) -> MedAIState:
        result = await self.recommender.generate(
            parsed_report=state.parsed_report,
            abnormal_values=state.abnormal_values,
            health_score=state.health_score,
        )
        state.recommendations = result["recommendations"]
        return state

    async def _node_finalize(self, state: MedAIState) -> MedAIState:
        logger.info(f"[Orchestrator] Workflow complete | Score: {state.health_score} | Emergency: {state.is_emergency}")
        return state

    def _route_emergency(self, state: MedAIState) -> str:
        return "emergency" if state.is_emergency else "normal"

    # ── Public API ─────────────────────────────
    async def run_report_analysis(self, user_id: str, file_path: str) -> Dict:
        state = MedAIState(user_id=user_id, input_type="report", raw_file_path=file_path)
        result = await self._report_graph.ainvoke(state)
        return {
            "summary":         result.ai_response,
            "parsed_report":   result.parsed_report,
            "abnormal_values": result.abnormal_values,
            "health_score":    result.health_score,
            "recommendations": result.recommendations,
            "is_emergency":    result.is_emergency,
            "severity":        result.severity,
        }

    async def run_chat(self, user_id: str, query: str, history: List[Dict]) -> Dict:
        state = MedAIState(
            user_id=user_id,
            input_type="chat",
            user_query=query,
            conversation_history=history,
        )
        result = await self._chat_graph.ainvoke(state)
        return {
            "response":    result.ai_response,
            "is_emergency": result.is_emergency,
            "severity":    result.severity,
            "sources":     result.retrieved_sources,
        }


orchestrator = AgentOrchestrator()
