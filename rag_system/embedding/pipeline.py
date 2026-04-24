"""
Embedding pipeline for generating vector representations.
Abstracts embedding model operations for easy swapping.
"""
from typing import List, Dict, Optional
import time
from abc import ABC, abstractmethod

from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

from ..core.types import Chunk
from ..core.config import EmbeddingConfig
from ..core.logging_utils import get_logger, timer


logger = get_logger("embedding")


class EmbeddingModel(ABC):
    """Abstract base for embedding models."""
    
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a batch of texts."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimension of embeddings."""
        pass


class HuggingFaceEmbeddingModel(EmbeddingModel):
    """HuggingFace-based embedding model."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        with timer(f"Loading embedding model: {model_name}"):
            self.embedder = HuggingFaceEmbeddings(model_name=model_name)
        
        # Determine dimension by embedding a dummy text
        dummy_embedding = self.embedder.embed_query("test")
        self._dimension = len(dummy_embedding)
        
        logger.info(
            f"Loaded embedding model {model_name} "
            f"({self._dimension} dimensions)"
        )
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text."""
        return np.array(self.embedder.embed_query(text))
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts efficiently."""
        embeddings = self.embedder.embed_documents(texts)
        return [np.array(e) for e in embeddings]
    
    @property
    def dimension(self) -> int:
        return self._dimension


class EmbeddingCache:
    """
    Simple in-memory cache for embeddings.
    Prevents re-embedding identical texts.
    
    Production: Consider using Redis or similar for distributed caching.
    """
    
    def __init__(self):
        self._cache: Dict[str, np.ndarray] = {}
        self._hits = 0
        self._misses = 0
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache if available."""
        if text in self._cache:
            self._hits += 1
            return self._cache[text]
        self._misses += 1
        return None
    
    def put(self, text: str, embedding: np.ndarray):
        """Store embedding in cache."""
        self._cache[text] = embedding
    
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    def stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'size': len(self._cache),
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self.hit_rate()
        }


class EmbeddingPipeline:
    """
    Manages embedding generation with batching, caching, and timing.
    
    Design rationale:
    - Abstracts embedding model for easy switching (GRANITE vs others)
    - Supports batching to amortize model loading overhead
    - Optional caching to avoid re-embedding identical chunks
    - Tracks latency for performance analysis
    """
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = HuggingFaceEmbeddingModel(config.model_name)
        self.cache = EmbeddingCache() if config.embed_cache else None
        self._embedding_times = []
        
        logger.info(
            f"Initialized embedding pipeline with {config.model_name} "
            f"(batch_size={config.batch_size})"
        )
    
    def embed_chunks(self, chunks: List[Chunk]) -> Dict[str, np.ndarray]:
        """
        Embed all chunks and return mapping of chunk_id -> embedding.
        
        Args:
            chunks: List of Chunk objects to embed
        
        Returns:
            Dictionary mapping chunk_id -> embedding vector
        """
        embeddings_map = {}
        texts_to_embed = []
        indices_to_embed = []
        
        with timer(f"Embedding {len(chunks)} chunks"):
            # First pass: check cache
            for i, chunk in enumerate(chunks):
                if self.cache is not None:
                    cached = self.cache.get(chunk.content)
                    if cached is not None:
                        embeddings_map[chunk.chunk_id] = cached
                        continue
                
                texts_to_embed.append(chunk.content)
                indices_to_embed.append(i)
            
            # Batch embedding for uncached texts
            if texts_to_embed:
                embeddings = self._embed_with_batching(texts_to_embed)
                
                for embedding, chunk_idx in zip(embeddings, indices_to_embed):
                    chunk = chunks[chunk_idx]
                    embeddings_map[chunk.chunk_id] = embedding
                    
                    # Cache the new embedding
                    if self.cache is not None:
                        self.cache.put(chunk.content, embedding)
            
            # Log stats
            logger.info(
                f"Embedded {len(embeddings_map)} chunks "
                f"(dimension={self.model.dimension})"
            )
            
            if self.cache is not None:
                cache_stats = self.cache.stats()
                logger.info(f"Cache stats: {cache_stats}")
        
        return embeddings_map
    
    def _embed_with_batching(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed texts with configurable batch size for efficiency.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embeddings in same order as input
        """
        all_embeddings = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            start_time = time.time()
            batch_embeddings = self.model.embed_batch(batch)
            elapsed_ms = (time.time() - start_time) * 1000
            
            self._embedding_times.append(elapsed_ms)
            all_embeddings.extend(batch_embeddings)
            
            logger.debug(
                f"Embedded batch {i//batch_size + 1} "
                f"({len(batch)} texts, {elapsed_ms:.2f}ms)"
            )
        
        return all_embeddings
    
    def get_avg_embedding_time(self) -> float:
        """Get average embedding time in milliseconds."""
        if not self._embedding_times:
            return 0.0
        return sum(self._embedding_times) / len(self._embedding_times)
