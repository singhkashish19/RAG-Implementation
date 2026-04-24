"""
EVALUATION EXPERIMENT SCRIPT
============================

This script runs comprehensive evaluation of the RAG system across different configurations.

WHAT IT DOES:
1. Load documents and create chunks with different chunk sizes
2. Generate synthetic evaluation dataset with ground truth
3. Evaluate retrieval quality across combinations of:
   - Chunk sizes: [256, 512, 1024] tokens
   - Top-k values: [3, 5, 10]
4. Generate comparison table and identify optimal configurations

EXPECTED OUTPUT:
- Evaluation results table
- Comparison of Recall@K, Precision@K, MRR across configurations
- Recommendations for production deployment

RUN:
    python evaluate_system.py
"""
import sys
import os
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rag_system.core import RAGConfig, ChunkingConfig, setup_logging, get_logger
from rag_system.ingestion import IngestionPipeline
from rag_system.embedding import EmbeddingPipeline
from rag_system.retrieval import SimpleVectorStore, RetrievalPipeline
from rag_system.evaluation import (
    EvaluationDatasetGenerator,
    SyntheticQueryStrategy,
    RetrieverEvaluator,
    ComparativeEvaluator,
)


logger = setup_logging()


class RAGEvaluationExperiment:
    """
    Orchestrates comprehensive RAG system evaluation.
    
    Workflow:
    1. Load and ingest documents
    2. Generate evaluation dataset
    3. Test across chunk size x top-k combinations
    4. Report results and recommendations
    """
    
    def __init__(self, document_path: str):
        self.document_path = document_path
        self.chunks = []
        self.eval_dataset = []
    
    def run_full_evaluation(self) -> ComparativeEvaluator:
        """
        Execute full evaluation pipeline.
        
        Returns:
            ComparativeEvaluator with all results
        """
        logger.info("=" * 80)
        logger.info("STARTING RAG SYSTEM EVALUATION EXPERIMENT")
        logger.info("=" * 80)
        
        # Step 1: Ingest documents
        logger.info("\n[STEP 1/4] Document Ingestion")
        self._ingest_documents()
        
        # Step 2: Generate evaluation dataset
        logger.info("\n[STEP 2/4] Synthetic Evaluation Dataset Generation")
        self._generate_eval_dataset()
        
        # Step 3: Run comparative evaluation
        logger.info("\n[STEP 3/4] Comparative Evaluation")
        evaluator = self._run_comparative_evaluation()
        
        # Step 4: Generate report
        logger.info("\n[STEP 4/4] Report Generation")
        self._print_results(evaluator)
        
        return evaluator
    
    def _ingest_documents(self):
        """Step 1: Load and chunk documents."""
        if not os.path.exists(self.document_path):
            logger.error(f"Document not found: {self.document_path}")
            raise FileNotFoundError(f"Document not found: {self.document_path}")
        
        # Use standard configuration
        config = RAGConfig()
        pipeline = IngestionPipeline(
            config=config.chunking,
            tokenizer_model=config.embedding.model_name
        )
        
        self.chunks = pipeline.ingest(self.document_path)
        logger.info(f"✓ Ingested {len(self.chunks)} chunks from {self.document_path}")
    
    def _generate_eval_dataset(self):
        """Step 2: Create synthetic evaluation queries."""
        strategy = SyntheticQueryStrategy(
            num_queries=20,  # Use smaller set for faster evaluation
            min_relevant_chunks=2,
            max_relevant_chunks=4,
            seed=42
        )
        
        generator = EvaluationDatasetGenerator(strategy)
        self.eval_dataset = generator.generate_dataset(self.chunks)
        
        logger.info(f"✓ Generated {len(self.eval_dataset)} evaluation queries")
        
        # Show sample queries
        logger.info("\nSample evaluation queries:")
        for i, query in enumerate(self.eval_dataset[:3], 1):
            logger.info(
                f"  {i}. Query: \"{query.query_text}\" "
                f"(expected {len(query.expected_chunk_ids)} chunks)"
            )
    
    def _run_comparative_evaluation(self) -> ComparativeEvaluator:
        """Step 3: Evaluate across configurations."""
        comparative_eval = ComparativeEvaluator()
        
        # Test configurations
        chunk_sizes = [256, 512, 1024]
        top_k_values = [3, 5, 10]
        
        total_configs = len(chunk_sizes) * len(top_k_values)
        config_num = 0
        
        # Base embedding config (same for all)
        base_config = RAGConfig()
        embedding_pipeline = EmbeddingPipeline(base_config.embedding)
        
        for chunk_size in chunk_sizes:
            logger.info(f"\n{'─' * 70}")
            logger.info(f"Evaluating chunk_size={chunk_size}")
            logger.info(f"{'─' * 70}")
            
            # Step A: Ingest with this chunk size
            config = RAGConfig()
            config.chunking.chunk_size = chunk_size
            
            ingestion_pipeline = IngestionPipeline(
                config=config.chunking,
                tokenizer_model=base_config.embedding.model_name
            )
            chunks = ingestion_pipeline.ingest(self.document_path)
            
            # Step B: Embed chunks
            start_embed = time.time()
            embeddings = embedding_pipeline.embed_chunks(chunks)
            embedding_time = time.time() - start_embed
            avg_embedding_time = embedding_time * 1000 / len(chunks)
            
            # Step C: Create retriever
            vector_store = SimpleVectorStore()
            vector_store.add_chunks(chunks, embeddings)
            
            # Step D: Evaluate across top-k values
            for top_k in top_k_values:
                config_num += 1
                logger.info(
                    f"  Config {config_num}/{total_configs}: "
                    f"chunk_size={chunk_size}, top_k={top_k}"
                )
                
                config.retrieval.top_k = top_k
                retriever = RetrievalPipeline(
                    config.retrieval,
                    embedding_pipeline.model.embedder,
                    vector_store
                )
                
                # Evaluate
                evaluator = RetrieverEvaluator(retriever)
                _, aggregate_metrics = evaluator.evaluate(
                    self.eval_dataset,
                    top_k=top_k
                )
                
                # Store results
                comparative_eval.add_result(
                    chunk_size=chunk_size,
                    top_k=top_k,
                    metrics=aggregate_metrics,
                    embedding_time_ms=avg_embedding_time
                )
                
                # Log this result
                logger.info(
                    f"    Recall@{top_k}: {aggregate_metrics['recall_at_k']:.4f}, "
                    f"Precision@{top_k}: {aggregate_metrics['precision_at_k']:.4f}, "
                    f"MRR: {aggregate_metrics['mrr']:.4f}, "
                    f"Latency: {aggregate_metrics['avg_latency_ms']:.2f}ms"
                )
        
        return comparative_eval
    
    @staticmethod
    def _print_results(evaluator: ComparativeEvaluator):
        """Step 4: Display comprehensive results."""
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 80)
        
        # Print comparison table
        logger.info("\n" + evaluator.get_comparison_table())
        
        # Print recommendations
        logger.info("\n" + "=" * 80)
        logger.info("RECOMMENDATIONS")
        logger.info("=" * 80)
        
        best_configs = evaluator.get_best_configurations()
        
        logger.info("\nOptimal configurations by objective:")
        logger.info(
            f"  • Best Recall: chunk_size={best_configs['best_recall'][0]}, "
            f"top_k={best_configs['best_recall'][1]}"
        )
        logger.info(
            f"  • Best Precision: chunk_size={best_configs['best_precision'][0]}, "
            f"top_k={best_configs['best_precision'][1]}"
        )
        logger.info(
            f"  • Best Latency: chunk_size={best_configs['best_latency'][0]}, "
            f"top_k={best_configs['best_latency'][1]}"
        )
        logger.info(
            f"  • Best Balanced: chunk_size={best_configs['best_balanced'][0]}, "
            f"top_k={best_configs['best_balanced'][1]}"
        )
        
        logger.info("\n" + evaluator.generate_report())


def main():
    """Main entry point for evaluation experiment."""
    # Find the State of Union document
    document_path = "state_of_the_union.txt"
    
    # Try to download if not present
    if not os.path.exists(document_path):
        logger.info("Downloading State of the Union document...")
        try:
            # Cross-platform download using urllib
            import urllib.request
            url = 'https://raw.githubusercontent.com/IBM/watson-machine-learning-samples/master/cloud/data/foundation_models/state_of_the_union.txt'
            urllib.request.urlretrieve(url, document_path)
            logger.info(f"\n✓ Downloaded to {document_path}")
        except Exception as e:
            logger.error(f"Failed to download: {e}")
            logger.error("Please provide state_of_the_union.txt in current directory")
            return
    
    # Add timeout and error handling for evaluation
    logger.info("Starting evaluation (this may take a few minutes)...")
    
    try:
        experiment = RAGEvaluationExperiment(document_path)
        evaluator = experiment.run_full_evaluation()
    except KeyboardInterrupt:
        logger.warning("\n⚠ Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"\n⚠ Evaluation failed: {e}")
        logger.info("Try running with smaller evaluation dataset or check system resources")
    
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
