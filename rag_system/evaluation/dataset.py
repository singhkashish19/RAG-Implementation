"""
Synthetic evaluation dataset generation.

Creates realistic evaluation queries with ground truth annotations
for measuring retrieval performance when no labeled dataset exists.
"""
from dataclasses import dataclass
from typing import List, Set, Optional
import random
import hashlib

from ..core.types import Chunk, EvaluationQuery
from ..core.logging_utils import get_logger, timer


logger = get_logger("evaluation_dataset")


@dataclass
class SyntheticQueryStrategy:
    """Strategy for generating synthetic evaluation queries."""
    num_queries: int = 50  # Number of evaluation queries to generate
    min_relevant_chunks: int = 2  # Min chunks to mark as relevant
    max_relevant_chunks: int = 5  # Max chunks to mark as relevant
    seed: int = 42  # Random seed for reproducibility


class EvaluationDatasetGenerator:
    """
    Generates synthetic evaluation datasets from document chunks.
    
    Strategy:
    1. Group chunks by source document
    2. For each document, select random chunks
    3. Extract key phrases from selected chunks as "queries"
    4. Mark related chunks as ground truth relevant
    
    This creates realistic retrieval evaluation scenarios without
    requiring manual annotation.
    """
    
    def __init__(self, strategy: SyntheticQueryStrategy = None):
        self.strategy = strategy or SyntheticQueryStrategy()
        random.seed(self.strategy.seed)
    
    def generate_dataset(self, chunks: List[Chunk]) -> List[EvaluationQuery]:
        """
        Generate evaluation queries from a list of chunks.
        
        Args:
            chunks: List of Chunk objects from ingestion pipeline
        
        Returns:
            List of EvaluationQuery objects with ground truth
        """
        with timer(f"Generating {self.strategy.num_queries} evaluation queries"):
            eval_queries = []
            
            # Group chunks by source
            chunks_by_source = self._group_chunks_by_source(chunks)
            
            logger.info(f"Chunks distributed across {len(chunks_by_source)} sources")
            
            # Generate queries
            all_chunks = chunks  # Keep reference to all chunks for selection
            
            for i in range(self.strategy.num_queries):
                # Randomly select a source document
                source = random.choice(list(chunks_by_source.keys()))
                source_chunks = chunks_by_source[source]
                
                # Randomly select relevant chunks from this source
                num_relevant = random.randint(
                    self.strategy.min_relevant_chunks,
                    min(self.strategy.max_relevant_chunks, len(source_chunks))
                )
                relevant_chunks = random.sample(source_chunks, num_relevant)
                relevant_chunk_ids = [c.chunk_id for c in relevant_chunks]
                
                # Generate query by extracting key phrases
                query_text = self._generate_query_from_chunks(relevant_chunks)
                
                eval_query = EvaluationQuery(
                    query_text=query_text,
                    expected_chunk_ids=relevant_chunk_ids,
                    metadata={
                        'source': source,
                        'num_relevant': num_relevant,
                        'strategy': 'synthetic_phrase_extraction'
                    }
                )
                eval_queries.append(eval_query)
            
            logger.info(f"Generated {len(eval_queries)} evaluation queries")
            
            return eval_queries
    
    @staticmethod
    def _group_chunks_by_source(chunks: List[Chunk]) -> dict[str, List[Chunk]]:
        """Group chunks by their source document."""
        grouped = {}
        for chunk in chunks:
            source = chunk.source
            if source not in grouped:
                grouped[source] = []
            grouped[source].append(chunk)
        return grouped
    
    @staticmethod
    def _generate_query_from_chunks(chunks: List[Chunk]) -> str:
        """
        Generate a query by extracting key phrases from chunks.
        
        Strategy:
        - Extract sentences from chunks
        - Combine sentences into a coherent query
        - This ensures the query is answerable by the selected chunks
        """
        # Combine text from chunks
        combined_text = " ".join([chunk.content for chunk in chunks])
        
        # Split into sentences (simple heuristic)
        sentences = combined_text.split('. ')
        
        # Take first 1-2 sentences
        query_sentences = random.sample(
            sentences[:max(1, len(sentences)//2)],
            k=min(2, len(sentences))
        )
        
        query_text = ". ".join(query_sentences).strip()
        
        # Ensure it looks like a question by appending a phrase
        query_phrases = [
            "What is mentioned about",
            "Tell me about",
            "Explain",
            "What are the details on",
            "How does this relate to",
        ]
        
        # Extract a key phrase (words that aren't too common)
        words = query_text.split()
        # Take first few meaningful words
        key_words = [w for w in words[:5] if len(w) > 4]
        
        if key_words:
            phrase = random.choice(query_phrases)
            query_text = f"{phrase}: {' '.join(key_words[:3])}?"
        
        return query_text
    
    @staticmethod
    def create_balanced_dataset(
        chunks: List[Chunk],
        num_queries: int = 50,
        seed: int = 42
    ) -> List[EvaluationQuery]:
        """
        Convenience method to create a balanced evaluation dataset.
        
        Args:
            chunks: Chunks from ingestion
            num_queries: Number of evaluation queries
            seed: Random seed for reproducibility
        
        Returns:
            List of evaluation queries
        """
        strategy = SyntheticQueryStrategy(
            num_queries=num_queries,
            seed=seed
        )
        generator = EvaluationDatasetGenerator(strategy)
        return generator.generate_dataset(chunks)


class ManualEvaluationDataset:
    """
    Builder for manually creating evaluation datasets.
    Useful for adding domain expertise to evaluation.
    """
    
    def __init__(self):
        self.queries: List[EvaluationQuery] = []
    
    def add_query(self, query_text: str, expected_chunk_ids: List[str]):
        """
        Manually add an evaluation query.
        
        Args:
            query_text: The query
            expected_chunk_ids: List of chunk IDs that should be retrieved
        """
        query = EvaluationQuery(
            query_text=query_text,
            expected_chunk_ids=expected_chunk_ids,
            metadata={'source': 'manual'}
        )
        self.queries.append(query)
        return self
    
    def build(self) -> List[EvaluationQuery]:
        """Return the built dataset."""
        return self.queries
