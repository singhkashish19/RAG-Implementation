"""
Quick Example: Using the Production RAG System

This script demonstrates:
1. System initialization
2. Querying with evaluation
3. Configuration customization
4. API usage

Run: python example_usage.py
"""
import os
from rag_system import RAGSystem
from rag_system.core import RAGConfig, RAGQuery, setup_logging

# Setup logging
logger = setup_logging()

def example_basic_query():
    """Example 1: Basic query."""
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 1: Basic Query")
    logger.info("="*80)
    
    # Initialize
    config = RAGConfig()
    document_path = "state_of_the_union.txt"
    
    if not os.path.exists(document_path):
        logger.error(f"Please provide {document_path}")
        return
    
    # Create RAG system
    try:
        rag = RAGSystem(
            config,
            document_path=document_path,
            llm_api_token=os.getenv("REPLICATE_API_TOKEN")
        )
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        logger.info("(This is OK for demo - showing structure)")
        return
    
    # Query
    response = rag.query(RAGQuery(
        query_text="What did the president say about climate change?"
    ))
    
    logger.info(f"\nQuery: {response.query}")
    logger.info(f"Answer: {response.answer[:200]}...")
    logger.info(f"Latency: {response.latency_ms:.2f}ms")
    logger.info(f"Retrieved: {len(response.retrieved_chunks)} chunks")


def example_custom_config():
    """Example 2: Custom configuration."""
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 2: Custom Configuration")
    logger.info("="*80)
    
    # Create custom config
    config = RAGConfig()
    config.chunking.chunk_size = 256  # Smaller chunks
    config.retrieval.top_k = 10  # More results
    
    logger.info(f"Chunk size: {config.chunking.chunk_size}")
    logger.info(f"Top-K: {config.retrieval.top_k}")
    logger.info(f"Model: {config.embedding.model_name}")
    
    # Show tradeoffs
    logger.info("\nTradeoff Analysis:")
    logger.info(config.get_tradeoff_summary())


def example_predefined_configs():
    """Example 3: Predefined configurations."""
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 3: Predefined Configurations")
    logger.info("="*80)
    
    from rag_system.core.config import (
        PRODUCTION_CONFIG,
        LATENCY_OPTIMIZED_CONFIG,
        QUALITY_OPTIMIZED_CONFIG,
    )
    
    configs = {
        "Production (Balanced)": PRODUCTION_CONFIG,
        "Latency Optimized": LATENCY_OPTIMIZED_CONFIG,
        "Quality Optimized": QUALITY_OPTIMIZED_CONFIG,
    }
    
    for name, config in configs.items():
        logger.info(f"\n{name}:")
        logger.info(f"  Chunk size: {config.chunking.chunk_size}")
        logger.info(f"  Top-K: {config.retrieval.top_k}")
        logger.info(f"  Reranking: {config.retrieval.enable_reranking}")


def example_evaluation():
    """Example 4: Evaluation metrics concepts."""
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 4: Understanding Evaluation Metrics")
    logger.info("="*80)
    
    from rag_system.evaluation import RetrievelMetricsCalculator
    
    # Simulate retrieval results
    relevant_ids = {"chunk_1", "chunk_5", "chunk_7"}  # Ground truth
    retrieved_ids = ["chunk_2", "chunk_5", "chunk_8", "chunk_1"]  # Got back these
    
    calculator = RetrievelMetricsCalculator()
    
    recall = calculator.recall_at_k(relevant_ids, retrieved_ids, k=3)
    precision = calculator.precision_at_k(relevant_ids, retrieved_ids, k=3)
    mrr = calculator.mean_reciprocal_rank(relevant_ids, retrieved_ids)
    
    logger.info(f"\nSimulated Retrieval:")
    logger.info(f"  Relevant chunks (ground truth): {relevant_ids}")
    logger.info(f"  Retrieved (top-3): {set(retrieved_ids[:3])}")
    logger.info(f"\nMetrics:")
    logger.info(f"  Recall@3: {recall:.2%} (2 of 3 relevant found)")
    logger.info(f"  Precision@3: {precision:.2%} (2 of 3 retrieved were relevant)")
    logger.info(f"  MRR: {mrr:.4f} (first relevant at rank 3)")


def main():
    """Run all examples."""
    logger.info("╔" + "="*78 + "╗")
    logger.info("║" + " "*20 + "RAG SYSTEM - USAGE EXAMPLES" + " "*33 + "║")
    logger.info("╚" + "="*78 + "╝")
    
    # Run examples
    example_basic_query()
    example_custom_config()
    example_predefined_configs()
    example_evaluation()
    
    logger.info("\n" + "="*80)
    logger.info("EXAMPLES COMPLETE")
    logger.info("="*80)
    logger.info("\nNext Steps:")
    logger.info("1. Run: python evaluate_system.py  (comprehensive evaluation)")
    logger.info("2. Deploy: See README.md for FastAPI setup")
    logger.info("3. Customize: Edit config.py in rag_system/core/")
    logger.info("\nFor full architecture details, see ARCHITECTURE.md")


if __name__ == "__main__":
    main()
