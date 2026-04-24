"""
Evaluation framework for retrieval quality.

CRITICAL: This module implements the quantitative evaluation pipeline.

Metrics:
- Recall@k: Proportion of relevant chunks that are retrieved in top-k
- Precision@k: Proportion of retrieved chunks that are relevant
- MRR: Mean Reciprocal Rank (position of first relevant result)
- Latency: Time to retrieve results

These metrics provide rigorous measurement of retrieval performance across
different chunk sizes and top-k values, enabling data-driven optimization.
"""
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional
import time
from collections import defaultdict

from ..core.types import Chunk, EvaluationQuery, EvaluationMetrics, RetrievalResult
from ..core.config import ChunkingConfig, RetrievalConfig
from ..core.logging_utils import get_logger, timer


logger = get_logger("evaluation")


@dataclass
class RetrievalEvaluationResult:
    """Results of evaluating retrieval on a single query."""
    query: str
    expected_chunk_ids: Set[str]
    retrieved_chunk_ids: List[str]  # Ranked list
    retrieved_chunks: List[RetrievalResult]
    
    # Metrics
    recall_at_k: float
    precision_at_k: float
    mrr: float
    latency_ms: float


class RetrievalMetricsCalculator:
    """
    Calculates retrieval quality metrics.
    
    Recall@K: Of the relevant docs, how many appear in top-K?
        - Range: 0-1
        - Higher is better
        - Formula: |relevant ∩ top_k| / |relevant|
    
    Precision@K: Of the top-K retrieved, how many are relevant?
        - Range: 0-1
        - Higher is better
        - Formula: |relevant ∩ top_k| / K
    
    MRR (Mean Reciprocal Rank): 1 / (rank of first relevant)
        - Range: 0-1
        - Higher is better
        - Useful for assessing ranking quality
    """
    
    @staticmethod
    def recall_at_k(
        relevant_ids: Set[str],
        retrieved_ids: List[str],
        k: int
    ) -> float:
        """
        Recall@K: Proportion of relevant items in top-K results.
        
        Args:
            relevant_ids: Set of relevant chunk IDs (ground truth)
            retrieved_ids: List of retrieved chunk IDs (ranked)
            k: Cutoff position
        
        Returns:
            Recall@K (0-1, higher better)
        """
        if not relevant_ids:
            return 0.0
        
        top_k_ids = set(retrieved_ids[:k])
        num_relevant_in_top_k = len(relevant_ids & top_k_ids)
        
        return num_relevant_in_top_k / len(relevant_ids)
    
    @staticmethod
    def precision_at_k(
        relevant_ids: Set[str],
        retrieved_ids: List[str],
        k: int
    ) -> float:
        """
        Precision@K: Proportion of relevant items among top-K results.
        
        Args:
            relevant_ids: Set of relevant chunk IDs
            retrieved_ids: List of retrieved chunk IDs (ranked)
            k: Cutoff position
        
        Returns:
            Precision@K (0-1, higher better)
        """
        top_k_ids = set(retrieved_ids[:k])
        num_relevant_in_top_k = len(relevant_ids & top_k_ids)
        
        return num_relevant_in_top_k / k if k > 0 else 0.0
    
    @staticmethod
    def mean_reciprocal_rank(
        relevant_ids: Set[str],
        retrieved_ids: List[str]
    ) -> float:
        """
        MRR: 1 / position of first relevant item.
        
        Args:
            relevant_ids: Set of relevant chunk IDs
            retrieved_ids: List of retrieved chunk IDs (ranked)
        
        Returns:
            MRR (0-1, higher better)
        """
        for rank, chunk_id in enumerate(retrieved_ids, 1):
            if chunk_id in relevant_ids:
                return 1.0 / rank
        
        # No relevant items found
        return 0.0


class RetrieverEvaluator:
    """
    Evaluates retriever performance on a set of evaluation queries.
    
    Produces comparison results across different chunk sizes and top-k values.
    """
    
    def __init__(self, retriever):
        """
        Args:
            retriever: RetrievalPipeline instance
        """
        self.retriever = retriever
        self.calculator = RetrievalMetricsCalculator()
    
    def evaluate(
        self,
        eval_queries: List[EvaluationQuery],
        top_k: int
    ) -> tuple[List[RetrievalEvaluationResult], Dict[str, float]]:
        """
        Evaluate retriever on a set of queries.
        
        Args:
            eval_queries: List of evaluation queries with ground truth
            top_k: Number of top results to retrieve
        
        Returns:
            (query_results, aggregate_metrics)
        """
        query_results = []
        total_latency = 0
        
        with timer(f"Evaluating {len(eval_queries)} queries (k={top_k})"):
            for eval_query in eval_queries:
                start_time = time.time()
                
                # Retrieve
                retrieved_results = self.retriever.retrieve(
                    query_text=eval_query.query_text,
                    top_k=top_k
                )
                
                latency_ms = (time.time() - start_time) * 1000
                total_latency += latency_ms
                
                # Extract chunk IDs
                retrieved_chunk_ids = [r.chunk.chunk_id for r in retrieved_results]
                expected_ids = set(eval_query.expected_chunk_ids)
                
                # Calculate metrics
                recall = self.calculator.recall_at_k(
                    expected_ids, retrieved_chunk_ids, top_k
                )
                precision = self.calculator.precision_at_k(
                    expected_ids, retrieved_chunk_ids, top_k
                )
                mrr = self.calculator.mean_reciprocal_rank(
                    expected_ids, retrieved_chunk_ids
                )
                
                result = RetrievalEvaluationResult(
                    query=eval_query.query_text,
                    expected_chunk_ids=expected_ids,
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    retrieved_chunks=retrieved_results,
                    recall_at_k=recall,
                    precision_at_k=precision,
                    mrr=mrr,
                    latency_ms=latency_ms
                )
                query_results.append(result)
        
        # Aggregate metrics
        avg_recall = sum(r.recall_at_k for r in query_results) / len(query_results)
        avg_precision = sum(r.precision_at_k for r in query_results) / len(query_results)
        avg_mrr = sum(r.mrr for r in query_results) / len(query_results)
        avg_latency = total_latency / len(query_results)
        
        aggregate_metrics = {
            'recall_at_k': avg_recall,
            'precision_at_k': avg_precision,
            'mrr': avg_mrr,
            'avg_latency_ms': avg_latency,
            'num_queries': len(query_results)
        }
        
        return query_results, aggregate_metrics


class ComparativeEvaluator:
    """
    Runs comprehensive evaluation across different configurations.
    
    Evaluates combinations of:
    - Chunk sizes (256, 512, 1024 tokens)
    - Top-k values (3, 5, 10)
    """
    
    def __init__(self):
        self.results: Dict[tuple, Dict] = {}
    
    def add_result(
        self,
        chunk_size: int,
        top_k: int,
        metrics: Dict[str, float],
        embedding_time_ms: float
    ):
        """Store evaluation result for a configuration."""
        key = (chunk_size, top_k)
        self.results[key] = {
            'metrics': metrics,
            'embedding_time_ms': embedding_time_ms
        }
    
    def get_comparison_table(self) -> str:
        """
        Generate human-readable comparison table.
        
        Returns comprehensive CSV-like output showing tradeoffs.
        """
        if not self.results:
            return "No results to compare"
        
        lines = []
        lines.append("=" * 100)
        lines.append(
            f"{'Chunk Size':>12} | {'Top-K':>6} | "
            f"{'Recall@K':>10} | {'Precision@K':>12} | "
            f"{'MRR':>8} | {'Latency(ms)':>12} | {'Embed(ms)':>10}"
        )
        lines.append("-" * 100)
        
        # Sort by chunk size, then top-k for readability
        for (chunk_size, top_k) in sorted(self.results.keys()):
            result = self.results[(chunk_size, top_k)]
            metrics = result['metrics']
            embed_time = result['embedding_time_ms']
            
            line = (
                f"{chunk_size:>12} | {top_k:>6} | "
                f"{metrics['recall_at_k']:>10.4f} | {metrics['precision_at_k']:>12.4f} | "
                f"{metrics['mrr']:>8.4f} | {metrics['avg_latency_ms']:>12.2f} | "
                f"{embed_time:>10.2f}"
            )
            lines.append(line)
        
        lines.append("=" * 100)
        return "\n".join(lines)
    
    def get_best_configurations(self) -> Dict:
        """Find configurations optimizing different objectives."""
        if not self.results:
            return {}
        
        # Best recall
        best_recall_key = max(
            self.results.keys(),
            key=lambda k: self.results[k]['metrics']['recall_at_k']
        )
        
        # Best precision
        best_precision_key = max(
            self.results.keys(),
            key=lambda k: self.results[k]['metrics']['precision_at_k']
        )
        
        # Best latency
        best_latency_key = min(
            self.results.keys(),
            key=lambda k: self.results[k]['metrics']['avg_latency_ms']
        )
        
        # Best balanced (F1-like score between recall and latency)
        best_balanced_key = max(
            self.results.keys(),
            key=lambda k: (
                self.results[k]['metrics']['recall_at_k'] -
                (self.results[k]['metrics']['avg_latency_ms'] / 100)
            )
        )
        
        return {
            'best_recall': best_recall_key,
            'best_precision': best_precision_key,
            'best_latency': best_latency_key,
            'best_balanced': best_balanced_key,
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive evaluation report."""
        report = [
            "\n" + "=" * 100,
            "RETRIEVAL SYSTEM EVALUATION REPORT",
            "=" * 100,
            "",
            "EVALUATION RESULTS",
            "-" * 100,
            self.get_comparison_table(),
            "",
            "ANALYSIS",
            "-" * 100,
        ]
        
        best_configs = self.get_best_configurations()
        
        if best_configs:
            report.append(f"Best Recall (chunk_size={best_configs['best_recall'][0]}, k={best_configs['best_recall'][1]})")
            report.append(f"Best Precision (chunk_size={best_configs['best_precision'][0]}, k={best_configs['best_precision'][1]})")
            report.append(f"Best Latency (chunk_size={best_configs['best_latency'][0]}, k={best_configs['best_latency'][1]})")
            report.append(f"Best Balanced (chunk_size={best_configs['best_balanced'][0]}, k={best_configs['best_balanced'][1]})")
        
        report.append("")
        report.append("=" * 100)
        
        return "\n".join(report)
