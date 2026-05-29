"""
MedAI Nexus — RAG Retrieval Agent
Complete RAG pipeline:
  Text → Chunking → Embedding → ChromaDB → Semantic Retrieval → Reranking
"""
from __future__ import annotations
import asyncio, hashlib, logging
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import CrossEncoder

from core.config import settings

logger = logging.getLogger("medai.rag_agent")

# Cross-encoder for reranking (runs locally — no extra cost)
_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RAGRetrievalAgent:
    """
    Autonomous RAG agent supporting:
    • Semantic search over ChromaDB
    • Metadata filtering (report type, user scope)
    • Cross-encoder reranking for precision
    • Hybrid retrieval (dense + sparse placeholder)
    """

    COLLECTION_MEDICAL = "medical_knowledge"   # static medical reference
    COLLECTION_REPORTS = "user_reports"        # per-user uploaded reports

    def __init__(self, embeddings: OpenAIEmbeddings):
        self.embeddings = embeddings
        self.splitter   = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=64,
            separators=["\n\n", "\n", ".", " "],
        )
        self._chroma = chromadb.HttpClient(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collections: Dict[str, chromadb.Collection] = {}
        try:
            self.reranker = CrossEncoder(_RERANKER_MODEL)
        except Exception:
            self.reranker = None
            logger.warning("[RAG] Reranker unavailable — skipping reranking step")

    # ── Ingest ─────────────────────────────────
    async def ingest(
        self,
        text: str,
        metadata: Dict[str, Any],
        collection: str = COLLECTION_REPORTS,
    ) -> List[str]:
        """
        Chunk text → generate embeddings → upsert into ChromaDB.
        Returns list of chunk IDs.
        """
        chunks = self.splitter.split_text(text)
        if not chunks:
            return []

        logger.info(f"[RAG] Ingesting {len(chunks)} chunks into '{collection}'")
        embeddings = await asyncio.to_thread(
            self.embeddings.embed_documents, chunks
        )
        col    = self._get_collection(collection)
        ids    = [self._chunk_id(text, i) for i in range(len(chunks))]
        metas  = [{**metadata, "chunk_index": i} for i in range(len(chunks))]

        await asyncio.to_thread(
            col.upsert,
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metas,
        )
        return ids

    # ── Retrieve ───────────────────────────────
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        collection: str = COLLECTION_MEDICAL,
        filter_metadata: Optional[Dict] = None,
    ) -> Dict:
        """
        Semantic search + optional cross-encoder reranking.
        Returns top-k chunks and source metadata.
        """
        logger.info(f"[RAG] Retrieving top-{top_k} chunks for query: '{query[:80]}...'")
        query_emb = await asyncio.to_thread(self.embeddings.embed_query, query)
        col       = self._get_collection(collection)

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_emb],
            "n_results": min(top_k * 2, 20),  # over-fetch then rerank
            "include": ["documents", "metadatas", "distances"],
        }
        if filter_metadata:
            kwargs["where"] = filter_metadata

        results = await asyncio.to_thread(col.query, **kwargs)

        docs    = results["documents"][0]
        metas   = results["metadatas"][0]
        scores  = results["distances"][0]

        # Cross-encoder reranking
        if self.reranker and len(docs) > 1:
            pairs        = [[query, doc] for doc in docs]
            rerank_scores = await asyncio.to_thread(self.reranker.predict, pairs)
            ranked       = sorted(zip(docs, metas, rerank_scores), key=lambda x: -x[2])
            docs, metas, scores = zip(*ranked)
            docs, metas, scores = list(docs), list(metas), list(scores)

        return {
            "chunks":  docs[:top_k],
            "sources": [m.get("source", "medical_reference") for m in metas[:top_k]],
            "scores":  scores[:top_k],
        }

    # ── Delete User Chunks ─────────────────────
    async def delete_user_chunks(self, user_id: str, report_id: str):
        col = self._get_collection(self.COLLECTION_REPORTS)
        await asyncio.to_thread(
            col.delete,
            where={"$and": [{"user_id": user_id}, {"report_id": report_id}]},
        )

    # ── Helpers ────────────────────────────────
    def _get_collection(self, name: str) -> chromadb.Collection:
        if name not in self._collections:
            self._collections[name] = self._chroma.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[name]

    @staticmethod
    def _chunk_id(text: str, index: int) -> str:
        h = hashlib.sha256(text[:64].encode()).hexdigest()[:12]
        return f"{h}_{index}"
