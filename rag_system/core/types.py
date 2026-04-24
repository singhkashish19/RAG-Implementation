"""
Core type definitions for the RAG system.
Provides common data structures used across all pipelines.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class Document:
    """Represents a source document in the knowledge base."""
    content: str
    doc_id: str
    source: str  # e.g., 'state_of_the_union.txt'
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __hash__(self):
        return hash(self.doc_id)


@dataclass
class Chunk:
    """Represents a text chunk (subdivision of a document)."""
    content: str
    chunk_id: str
    doc_id: str
    chunk_index: int  # Position within the document
    tokens: int  # Token count for this chunk
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.chunk_id)


@dataclass
class RetrievalResult:
    """Result from retrieval: chunk + relevance score."""
    chunk: Chunk
    score: float  # Similarity score (0-1)
    rank: int  # Rank in retrieval results


@dataclass
class RAGQuery:
    """User query input to the RAG system."""
    query_text: str
    top_k: int = 5
    user_id: Optional[str] = None
    request_id: Optional[str] = None


@dataclass
class RAGResponse:
    """Complete response from RAG system."""
    query: str
    answer: str
    retrieved_chunks: List[RetrievalResult]
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationMetrics:
    """Metrics for evaluating retrieval quality."""
    chunk_size: int
    top_k: int
    recall_at_k: float
    precision_at_k: float
    mrr: float  # Mean Reciprocal Rank
    latency_ms: float
    avg_embedding_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return (f"EvaluationMetrics("
                f"chunk_size={self.chunk_size}, "
                f"k={self.top_k}, "
                f"recall={self.recall_at_k:.4f}, "
                f"precision={self.precision_at_k:.4f}, "
                f"mrr={self.mrr:.4f}, "
                f"latency={self.latency_ms:.2f}ms)")


@dataclass
class EvaluationQuery:
    """Query with ground truth for evaluation."""
    query_text: str
    expected_chunk_ids: List[str]  # Ground truth relevant chunks
    metadata: Dict[str, Any] = field(default_factory=dict)
