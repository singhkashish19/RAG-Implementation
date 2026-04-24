"""
Retrieval pipeline for finding relevant chunks given a query.
Includes optional reranking for improved precision.
"""
from typing import List, Dict, Optional, Tuple
import time
import numpy as np
from abc import ABC, abstractmethod

from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings

from ..core.types import Chunk, RetrievalResult, RAGQuery
from ..core.config import RetrievalConfig, EmbeddingConfig
from ..core.logging_utils import get_logger, timer


logger = get_logger("retrieval")


class VectorStore(ABC):
    """Abstract base for vector stores."""
    
    @abstractmethod
    def add_chunks(self, chunks: List[Chunk], embeddings: Dict[str, np.ndarray]):
        """Add chunks with embeddings to the store."""
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[Chunk, float]]:
        """Search for similar chunks. Returns (chunk, score) tuples."""
        pass


class MilvusVectorStore(VectorStore):
    """Wrapper around Milvus vector database."""
    
    def __init__(self, embedding_model: HuggingFaceEmbeddings, 
                 db_path: Optional[str] = None):
        self.embedding_model = embedding_model
        self.db_path = db_path or "milvus_rag.db"
        
        with timer(f"Initializing Milvus at {self.db_path}"):
            self.milvus = Milvus(
                embedding_function=embedding_model,
                connection_args={"uri": self.db_path},
                auto_id=True,
                index_params={"index_type": "AUTOINDEX"},
            )
        
        logger.info(f"Initialized Milvus vector store at {self.db_path}")
    
    def add_chunks(self, chunks: List[Chunk], embeddings: Dict[str, np.ndarray]):
        """Add chunks to Milvus (using LangChain integration)."""
        # LangChain's add_documents handles the embedding
        self.milvus.add_documents(chunks)
        logger.info(f"Added {len(chunks)} chunks to Milvus")
    
    def search(self, query_text: str, top_k: int) -> List[Tuple[Chunk, float]]:
        """
        Search Milvus using text query.
        
        LangChain's Milvus wrapper searches by text, which gets embedded
        automatically. Returns documents with similarity scores.
        
        Args:
            query_text: The search query text
            top_k: Number of results to return
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        try:
            # Use LangChain's similarity search with score
            docs_with_scores = self.milvus.similarity_search_with_score(
                query=query_text,
                k=top_k
            )
            
            results = []
            for doc, score in docs_with_scores:
                # LangChain document can be converted to Chunk
                if hasattr(doc, 'metadata') and 'chunk_id' in doc.metadata:
                    chunk = Chunk(
                        content=doc.page_content,
                        chunk_id=doc.metadata.get('chunk_id', ''),
                        doc_id=doc.metadata.get('doc_id', ''),
                        chunk_index=doc.metadata.get('chunk_index', 0),
                        tokens=doc.metadata.get('tokens', 0),
                        source=doc.metadata.get('source', ''),
                        metadata=doc.metadata
                    )
                    results.append((chunk, score))
                else:
                    # Fallback: create chunk from document
                    chunk = Chunk(
                        content=doc.page_content,
                        chunk_id=f"milvus_{len(results)}",
                        doc_id=doc.metadata.get('source', 'unknown'),
                        chunk_index=len(results),
                        tokens=len(doc.page_content.split()),
                        source=doc.metadata.get('source', 'unknown'),
                        metadata=doc.metadata
                    )
                    results.append((chunk, score))
            
            logger.info(f"Milvus search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Milvus search failed: {e}")
            return []


class SimpleVectorStore(VectorStore):
    """
    Simple in-memory vector store using cosine similarity.
    Useful for testing and small datasets.
    """
    
    def __init__(self):
        self.chunks: Dict[str, Chunk] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        logger.info("Initialized simple in-memory vector store")
    
    def add_chunks(self, chunks: List[Chunk], embeddings: Dict[str, np.ndarray]):
        """Store chunks and embeddings in memory."""
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk
            if chunk.chunk_id in embeddings:
                self.embeddings[chunk.chunk_id] = embeddings[chunk.chunk_id]
        
        logger.info(f"Added {len(chunks)} chunks to in-memory store")
    
    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[Chunk, float]]:
        """
        Find top-k similar chunks using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
        
        Returns:
            List of (chunk, similarity_score) tuples, sorted by score descending
        """
        scores = []
        
        for chunk_id, chunk_embedding in self.embeddings.items():
            # Cosine similarity
            similarity = np.dot(query_embedding, chunk_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding) + 1e-8
            )
            scores.append((chunk_id, similarity))
        
        # Sort by score descending, take top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = [
            (self.chunks[chunk_id], score)
            for chunk_id, score in scores[:top_k]
        ]
        
        return results


class Reranker(ABC):
    """Abstract base for reranking models."""
    
    @abstractmethod
    def rerank(self, query: str, candidates: List[Chunk]) -> List[Tuple[Chunk, float]]:
        """Rerank candidates and return sorted list with scores."""
        pass


class CrossEncoderReranker(Reranker):
    """
    Reranking using cross-encoder model.
    
    Cross-encoders directly score query-document pairs, giving better precision
    than embedding-based retrieval. Used as second-stage ranker.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
            logger.info(f"Loaded cross-encoder reranker: {model_name}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Reranking disabled. Install: pip install sentence-transformers"
            )
            self.model = None
    
    def rerank(self, query: str, candidates: List[Chunk]) -> List[Tuple[Chunk, float]]:
        """
        Score and rank candidates for the query.
        
        Args:
            query: Query text
            candidates: List of candidate chunks
        
        Returns:
            List of (chunk, score) sorted by score descending
        """
        if self.model is None:
            logger.warning("Reranker not available, returning unchanged order")
            return [(c, 1.0) for c in candidates]
        
        with timer(f"Reranking {len(candidates)} candidates"):
            # Score all pairs (query, candidate)
            pairs = [(query, chunk.content) for chunk in candidates]
            scores = self.model.predict(pairs)
            
            # Combine chunks with scores and sort
            results = sorted(
                zip(candidates, scores),
                key=lambda x: x[1],
                reverse=True
            )
        
        return results


class RetrievalPipeline:
    """
    Main retrieval pipeline orchestrating search and optional reranking.
    
    Workflow:
    1. Embed query
    2. Search vector store (fast approximate search)
    3. [Optional] Rerank top results (slower but more accurate)
    """
    
    def __init__(self, config: RetrievalConfig, embedding_model: HuggingFaceEmbeddings,
                 vector_store: VectorStore):
        self.config = config
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        
        self.reranker = None
        if config.enable_reranking and config.reranker_model:
            self.reranker = CrossEncoderReranker(config.reranker_model)
        
        self._retrieval_times = []
        logger.info(f"Initialized retrieval pipeline (k={config.top_k})")
    
    def retrieve(self, query_text: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query_text: Query text
            top_k: Override config top_k (useful for evaluation)
        
        Returns:
            List of RetrievalResult objects
        """
        if top_k is None:
            top_k = self.config.top_k
        
        start_time = time.time()
        
        with timer(f"Retrieving top-{top_k} chunks"):
            # Step 1: Embed query
            query_embedding = np.array(
                self.embedding_model.embed_query(query_text)
            )
            
            # Step 2: Vector search (approximate)
            candidates = self.vector_store.search(query_embedding, top_k=top_k * 2)
            
            # Step 3: Rerank if enabled
            if self.reranker is not None:
                candidate_chunks = [chunk for chunk, _ in candidates]
                reranked = self.reranker.rerank(query_text, candidate_chunks)
                candidates = reranked[:top_k]
            else:
                candidates = candidates[:top_k]
            
            # Convert to RetrievalResult objects
            results = [
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    rank=rank
                )
                for rank, (chunk, score) in enumerate(candidates, 1)
            ]
        
        elapsed_ms = (time.time() - start_time) * 1000
        self._retrieval_times.append(elapsed_ms)
        
        logger.debug(f"Retrieved {len(results)} chunks in {elapsed_ms:.2f}ms")
        
        return results
    
    def get_avg_retrieval_time(self) -> float:
        """Get average retrieval time in milliseconds."""
        if not self._retrieval_times:
            return 0.0
        return sum(self._retrieval_times) / len(self._retrieval_times)
