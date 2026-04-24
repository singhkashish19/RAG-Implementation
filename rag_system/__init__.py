"""
Main RAG system orchestrator.

Ties together all components (ingestion, embedding, retrieval) into
a cohesive, production-ready system.
"""
from typing import List, Optional
import time

from langchain_community.llms import Replicate
from transformers import AutoTokenizer

from .core import RAGConfig, RAGQuery, RAGResponse, get_logger, timer
from .ingestion import IngestionPipeline
from .embedding import EmbeddingPipeline
from .retrieval import SimpleVectorStore, RetrievalPipeline
from .core.logging_utils import timeit


logger = get_logger("rag_system")


class RAGSystem:
    """
    Complete RAG system: load documents, embed, retrieve, answer.
    
    Configuration:
    - Modular: Each component can be replaced
    - Testable: Components have clear interfaces
    - Observable: Latency tracking and logging
    
    Usage:
        config = RAGConfig()
        rag = RAGSystem(config, document_path="data.txt")
        response = rag.query("What about...", top_k=5)
        print(response.answer)
    """
    
    def __init__(self, config: RAGConfig, document_path: str, 
                 llm_api_token: Optional[str] = None):
        """
        Initialize RAG system.
        
        Args:
            config: RAGConfig with all component configurations
            document_path: Path to document(s) to index
            llm_api_token: API token for LLM (e.g., Replicate)
        """
        self.config = config
        self.document_path = document_path
        self.llm_api_token = llm_api_token
        
        logger.info("Initializing RAG system...")
        logger.info(config.get_tradeoff_summary())
        
        # Initialize components
        with timer("RAG System initialization"):
            self.ingestion_pipeline = IngestionPipeline(
                config.chunking,
                config.embedding.model_name
            )
            
            self.embedding_pipeline = EmbeddingPipeline(
                config.embedding
            )
            
            self.vector_store = SimpleVectorStore()
            
            self.retrieval_pipeline = RetrievalPipeline(
                config.retrieval,
                self.embedding_pipeline.model.embedder,
                self.vector_store
            )
            
            self.llm = self._initialize_llm()
            
            # Load and index documents
            self._index_documents()
        
        logger.info("✓ RAG system ready")
    
    def _initialize_llm(self):
        """Initialize LLM for answer generation."""
        try:
            logger.info(f"Loading LLM: {self.config.llm_model}")
            
            llm = Replicate(
                model=self.config.llm_model,
                replicate_api_token=self.llm_api_token,
            )
            
            logger.info("✓ LLM loaded")
            return llm
        
        except Exception as e:
            logger.warning(f"Failed to load LLM: {e}")
            logger.warning("LLM operations will fail until configured properly")
            return None
    
    def _index_documents(self):
        """Load and index documents into vector store."""
        logger.info(f"Indexing documents from {self.document_path}")
        
        # Ingest documents
        chunks = self.ingestion_pipeline.ingest(self.document_path)
        logger.info(f"Ingested {len(chunks)} chunks")
        
        # Embed chunks
        embeddings = self.embedding_pipeline.embed_chunks(chunks)
        logger.info(f"Embedded {len(embeddings)} chunks")
        
        # Add to vector store
        self.vector_store.add_chunks(chunks, embeddings)
        logger.info("✓ Documents indexed")
    
    @timeit
    def query(self, rag_query: RAGQuery) -> RAGResponse:
        """
        Process a query and return answer with retrieved context.
        
        Args:
            rag_query: RAGQuery with query text and parameters
        
        Returns:
            RAGResponse with answer and retrieved chunks
        """
        start_time = time.time()
        
        logger.info(f"Processing query: {rag_query.query_text[:50]}...")
        
        # Step 1: Retrieve relevant chunks
        retrieved_chunks = self.retrieval_pipeline.retrieve(
            rag_query.query_text,
            top_k=rag_query.top_k
        )
        
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
        
        # Step 2: Format context from retrieved chunks
        context = self._format_context(retrieved_chunks)
        
        # Step 3: Generate answer using LLM
        answer = self._generate_answer(rag_query.query_text, context)
        
        # Step 4: Compile response
        latency_ms = (time.time() - start_time) * 1000
        
        response = RAGResponse(
            query=rag_query.query_text,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            latency_ms=latency_ms,
            metadata={
                'config': {
                    'chunk_size': self.config.chunking.chunk_size,
                    'top_k': rag_query.top_k,
                    'embedding_model': self.config.embedding.model_name,
                    'llm_model': self.config.llm_model,
                }
            }
        )
        
        logger.info(
            f"Query complete: {len(retrieved_chunks)} chunks, "
            f"{latency_ms:.2f}ms latency"
        )
        
        return response
    
    @staticmethod
    def _format_context(retrieved_chunks: List) -> str:
        """Format retrieved chunks into context string."""
        context_lines = [
            "Retrieved Context:",
            "─" * 60
        ]
        
        for result in retrieved_chunks:
            context_lines.append(
                f"[Chunk {result.rank} | Relevance: {result.score:.2f}]"
            )
            context_lines.append(f"{result.chunk.content}")
            context_lines.append("")
        
        return "\n".join(context_lines)
    
    def _generate_answer(self, query: str, context: str) -> str:
        """
        Generate answer using LLM with retrieved context.
        
        Args:
            query: User query
            context: Retrieved context
        
        Returns:
            Generated answer
        """
        if self.llm is None:
            return (
                "[LLM not available] "
                "Retrieved context:\n" + context
            )
        
        prompt = self._build_prompt(query, context)
        
        try:
            logger.debug("Generating answer with LLM...")
            # Use the Replicate LLM wrapper correctly; it exposes invoke/generate rather than being directly callable
            answer = self.llm.invoke(prompt)
            return answer
        
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"Error generating answer: {e}"
    
    @staticmethod
    def _build_prompt(query: str, context: str) -> str:
        """
        Build prompt for LLM following Granite template.
        
        Args:
            query: User query
            context: Retrieved context
        
        Returns:
            Formatted prompt
        """
        return f"""You are a helpful assistant. Answer the following question based on the provided context.

Question: {query}

Context:
{context}

Answer:"""
    
    def get_system_info(self) -> dict:
        """Get system information for diagnostics."""
        return {
            'chunk_size': self.config.chunking.chunk_size,
            'chunk_overlap': self.config.chunking.chunk_overlap,
            'embedding_model': self.config.embedding.model_name,
            'embedding_batch_size': self.config.embedding.batch_size,
            'retrieval_top_k': self.config.retrieval.top_k,
            'retrieval_reranking': self.config.retrieval.enable_reranking,
            'llm_model': self.config.llm_model,
            'llm_temperature': self.config.llm_temperature,
            'vector_store_type': 'simple_in_memory',
        }


if __name__ == "__main__":
    # Example usage
    import os
    
    # Set up logging
    logger_instance = get_logger("main")
    
    # Configuration
    config = RAGConfig()
    
    # Create RAG system
    document_path = "state_of_the_union.txt"
    if not os.path.exists(document_path):
        logger_instance.error(f"Document not found: {document_path}")
    else:
        rag = RAGSystem(config, document_path)
        
        # Example query
        query = RAGQuery(query_text="What did the president say about Ketanji Brown Jackson?")
        response = rag.query(query)
        
        logger_instance.info("\n" + "=" * 80)
        logger_instance.info("QUERY RESPONSE")
        logger_instance.info("=" * 80)
        logger_instance.info(f"\nQuery: {response.query}")
        logger_instance.info(f"\nAnswer:\n{response.answer}")
        logger_instance.info(f"\nLatency: {response.latency_ms:.2f}ms")
        logger_instance.info(f"Retrieved: {len(response.retrieved_chunks)} chunks")
