"""
MedAI Nexus — Chat Routes
REST + WebSocket endpoints for AI medical chat
"""
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import asyncio, logging, json

from agents.orchestrator import orchestrator
from middleware.auth import get_current_user
from models.models import User, Conversation, Message
from core.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger("medai.chat")
router = APIRouter()


class ChatMessage(BaseModel):
    content: str
    conversation_id: Optional[str] = None


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    is_emergency: bool = False
    sources: List[str] = []


# ─── REST: Send Message ───────────────────────
@router.post("/message", response_model=MessageOut)
async def send_message(
    payload: ChatMessage,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Non-streaming chat endpoint.
    Runs full LangGraph chat workflow: triage → RAG → LLM → response.
    """
    # Load conversation history
    history = []
    if payload.conversation_id:
        conv = await _load_conversation(db, payload.conversation_id, current_user.id)
        history = [{"role": m.role, "content": m.content} for m in conv.messages[-12:]]

    # Run AI workflow
    result = await orchestrator.run_chat(
        user_id=current_user.id,
        query=payload.content,
        history=history,
    )

    # Persist messages
    if payload.conversation_id:
        await _save_messages(db, payload.conversation_id, [
            {"role": "user",      "content": payload.content},
            {"role": "assistant", "content": result["response"],
             "is_emergency": result["is_emergency"]},
        ])

    return MessageOut(
        id="msg_" + __import__("uuid").uuid4().hex[:8],
        role="assistant",
        content=result["response"],
        is_emergency=result["is_emergency"],
        sources=result.get("sources", []),
    )


# ─── REST: Conversation History ───────────────
@router.get("/conversations")
async def list_conversations(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import select
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc())
        .limit(30)
    )
    convs = result.scalars().all()
    return [{"id": c.id, "title": c.title, "updated_at": c.updated_at} for c in convs]


# ─── WebSocket: Streaming Chat ────────────────
@router.websocket("/ws/{conversation_id}")
async def websocket_chat(
    websocket: WebSocket,
    conversation_id: str,
):
    """
    WebSocket endpoint for real-time streaming AI responses.
    Streams token-by-token as they are generated.

    Protocol:
      Client → { "content": "<user message>", "token": "<jwt>" }
      Server → { "type": "token",  "data": "<chunk>" }  (streaming)
      Server → { "type": "done",   "is_emergency": bool }
      Server → { "type": "error",  "data": "<message>" }
    """
    await websocket.accept()
    logger.info(f"[WS] Connection opened for conv={conversation_id}")

    try:
        while True:
            raw  = await websocket.receive_text()
            data = json.loads(raw)
            query = data.get("content", "").strip()
            if not query:
                continue

            history = data.get("history", [])
            is_emergency = False

            try:
                # Stream response token by token
                async for chunk in _stream_response(query, history):
                    if chunk["type"] == "emergency":
                        is_emergency = True
                    await websocket.send_json(chunk)

                await websocket.send_json({"type": "done", "is_emergency": is_emergency})

            except Exception as e:
                logger.error(f"[WS] AI error: {e}")
                await websocket.send_json({"type": "error", "data": str(e)})

    except WebSocketDisconnect:
        logger.info(f"[WS] Client disconnected from conv={conversation_id}")


async def _stream_response(query: str, history: list):
    """Generator that yields streaming tokens from LLM."""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    from core.config import settings
    from agents.emergency_agent import EmergencyDetectionAgent

    # Emergency triage first (fast, keyword-based)
    emrg_agent = EmergencyDetectionAgent(None)
    is_emergency = emrg_agent.fast_triage(query)
    if is_emergency:
        yield {"type": "emergency", "data": ""}

    llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.OPENAI_API_KEY,
        streaming=True,
        temperature=0.3,
    )
    messages = [
        SystemMessage(content=(
            "You are MedAI, an expert AI healthcare assistant. Be empathetic, accurate, and clear. "
            "Use bullet points for structured answers. Always recommend consulting a licensed physician."
            + ("\n\n⚠️ EMERGENCY DETECTED — prioritise urgent care instructions." if is_emergency else "")
        )),
        *[HumanMessage(content=m["content"]) if m["role"] == "user"
          else SystemMessage(content=m["content"])
          for m in history[-8:]],
        HumanMessage(content=query),
    ]
    async for chunk in llm.astream(messages):
        if chunk.content:
            yield {"type": "token", "data": chunk.content}


async def _load_conversation(db, conv_id: str, user_id: str) -> Conversation:
    from sqlalchemy import select
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conv_id,
            Conversation.user_id == user_id
        )
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


async def _save_messages(db, conv_id: str, messages: list):
    import uuid
    for m in messages:
        msg = Message(
            id=str(uuid.uuid4()),
            conversation_id=conv_id,
            role=m["role"],
            content=m["content"],
            is_emergency=m.get("is_emergency", False),
        )
        db.add(msg)
    await db.commit()
