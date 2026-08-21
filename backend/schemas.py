from pydantic import BaseModel
from typing import Optional


class ChunkSource(BaseModel):
    chunk_id: str
    text: str
    strategy: str
    score: float
    passage_id: Optional[str] = None


class LatencyBreakdown(BaseModel):
    stt_ms: float
    classify_ms: float
    retrieve_ms: float
    rerank_ms: float
    guardrail_ms: float
    generate_ms: float
    total_ms: float


class RAGResponse(BaseModel):
    answer: str
    sources: list[ChunkSource]
    strategy_used: list[str]
    confidence: float
    grounded: bool
    latency_breakdown: LatencyBreakdown
    query_language: str
    query_text: str