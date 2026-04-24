"""
FastAPI deployment layer for the RAG system.

Exposes the RAG system via REST API with request logging and latency tracking.

ENDPOINTS:
- POST /query: Submit a query and get answer + context
- GET /health: Health check
- GET /metrics: System metrics

USAGE:
    from fastapi import FastAPI
    from rag_server import create_app
    
    app = create_app()
    
    # Run with: uvicorn rag_server:app --reload --port 8000
"""
from typing import List, Optional
from datetime import datetime
import time
import logging
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..core.types import RAGResponse, RAGQuery, RetrievalResult
from ..core.logging_utils import get_logger


logger = get_logger("api")


# Request/Response models
class QueryRequest(BaseModel):
    """Request model for RAG query."""
    query: str
    top_k: int = 5
    user_id: Optional[str] = None


class RetrievedChunkInfo(BaseModel):
    """Information about a retrieved chunk."""
    content: str
    source: str
    chunk_id: str
    relevance_score: float
    rank: int


class QueryResponse(BaseModel):
    """Response model for RAG query."""
    query: str
    answer: str
    retrieved_chunks: List[RetrievedChunkInfo]
    latency_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    model_loaded: bool


class MetricsResponse(BaseModel):
    """System metrics response."""
    total_queries_processed: int
    avg_latency_ms: float
    total_uptime_seconds: float
    embedding_model: str


class RequestLogger:
    """Logs all API requests for monitoring."""
    
    def __init__(self):
        self.requests = []
        self.total_queries = 0
        self.total_latency = 0
        self.start_time = time.time()
    
    def log_request(self, query: str, latency_ms: float, status: str = "success"):
        """Log a request."""
        self.requests.append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'latency_ms': latency_ms,
            'status': status
        })
        self.total_queries += 1
        self.total_latency += latency_ms
        
        logger.info(
            f"[REQUEST] query=\"{query[:50]}...\" "
            f"latency={latency_ms:.2f}ms status={status}"
        )
    
    def get_avg_latency(self) -> float:
        """Get average latency."""
        if self.total_queries == 0:
            return 0.0
        return self.total_latency / self.total_queries
    
    def get_uptime_seconds(self) -> float:
        """Get uptime in seconds."""
        return time.time() - self.start_time


def create_app(rag_system=None):
    """
    Create FastAPI application with RAG system.
    
    Args:
        rag_system: Initialized RAG system (if None, will be lazy-loaded)
    
    Returns:
        FastAPI app ready to serve
    """
    app = FastAPI(
        title="RAG System API",
        description="Production-grade Retrieval-Augmented Generation API",
        version="1.0.0"
    )
    
    # Add CORS middleware for cross-origin requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize request logger
    request_logger = RequestLogger()
    
    # Store RAG system as app state
    app.state.rag_system = rag_system
    app.state.request_logger = request_logger
    
    @app.middleware("http")
    async def add_request_tracking(request: Request, call_next):
        """Add request tracking middleware."""
        request.state.start_time = time.time()
        response = await call_next(request)
        return response
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """
        Health check endpoint.
        
        Returns system status and model loading status.
        """
        rag_ready = app.state.rag_system is not None
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            model_loaded=rag_ready
        )
    
    @app.post("/query", response_model=QueryResponse)
    async def process_query(request: QueryRequest):
        """
        Process a query and return answer with retrieved context.
        
        Args:
            request: QueryRequest with query text and optional top_k
        
        Returns:
            QueryResponse with answer and retrieved chunks
        
        Raises:
            HTTPException: If RAG system not ready or query fails
        """
        if app.state.rag_system is None:
            raise HTTPException(
                status_code=503,
                detail="RAG system not initialized. Please check server logs."
            )
        
        start_time = time.time()
        
        try:
            # Create RAG query
            rag_query = RAGQuery(
                query_text=request.query,
                top_k=request.top_k,
                user_id=request.user_id
            )
            
            # Process query through RAG system
            rag_response = app.state.rag_system.query(rag_query)
            
            # Convert to API response
            retrieved_chunks = [
                RetrievedChunkInfo(
                    content=result.chunk.content,
                    source=result.chunk.source,
                    chunk_id=result.chunk.chunk_id,
                    relevance_score=result.score,
                    rank=result.rank
                )
                for result in rag_response.retrieved_chunks
            ]
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Log request
            app.state.request_logger.log_request(
                query=request.query,
                latency_ms=latency_ms,
                status="success"
            )
            
            return QueryResponse(
                query=request.query,
                answer=rag_response.answer,
                retrieved_chunks=retrieved_chunks,
                latency_ms=latency_ms,
                timestamp=datetime.now().isoformat()
            )
        
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            app.state.request_logger.log_request(
                query=request.query,
                latency_ms=latency_ms,
                status="error"
            )
            
            logger.error(f"Query processing failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Query processing failed: {str(e)}"
            )
    
    @app.get("/metrics", response_model=MetricsResponse)
    async def get_metrics():
        """
        Get system metrics.
        
        Returns:
            MetricsResponse with performance statistics
        """
        logger = app.state.request_logger
        embedding_model = "unknown"
        
        if app.state.rag_system:
            try:
                embedding_model = app.state.rag_system.embedding_pipeline.config.model_name
            except:
                pass
        
        return MetricsResponse(
            total_queries_processed=logger.total_queries,
            avg_latency_ms=logger.get_avg_latency(),
            total_uptime_seconds=logger.get_uptime_seconds(),
            embedding_model=embedding_model
        )
    
    return app


if __name__ == "__main__":
    # Development entry point
    import uvicorn
    
    app = create_app()
    
    logger.info("Starting RAG API server...")
    logger.info("Visit http://localhost:8000/docs for interactive API documentation")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )
