"""
Document ingestion pipeline.
Handles loading, cleaning, and chunking of documents.
"""
from pathlib import Path
from typing import List, Optional, Dict, Any
import hashlib
from abc import ABC, abstractmethod

from langchain_community.document_loaders import TextLoader
from transformers import AutoTokenizer

from ..core.types import Document, Chunk
from ..core.config import ChunkingConfig
from ..core.logging_utils import get_logger, timer


logger = get_logger("ingestion")


class DocumentLoader(ABC):
    """Base class for document loaders."""
    
    @abstractmethod
    def load(self) -> List[Document]:
        """Load documents from source."""
        pass


class TextFileLoader(DocumentLoader):
    """Load text documents from files."""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
    
    def load(self) -> List[Document]:
        """Load a single text file as a document."""
        with timer(f"Loading document from {self.file_path}"):
            loader = TextLoader(str(self.file_path), encoding='utf-8')
            langchain_docs = loader.load()
            
            documents = []
            for langchain_doc in langchain_docs:
                doc = Document(
                    content=langchain_doc.page_content,
                    doc_id=self._generate_doc_id(langchain_doc.page_content),
                    source=str(self.file_path),
                    metadata=langchain_doc.metadata or {}
                )
                documents.append(doc)
                logger.info(f"Loaded document: {doc.doc_id} from {self.file_path}")
            
            return documents
    
    @staticmethod
    def _generate_doc_id(content: str) -> str:
        """Generate deterministic doc ID from content."""
        return hashlib.md5(content.encode()).hexdigest()[:12]


class DocumentChunker:
    """
    Splits documents into chunks using token-aware splitting.
    
    Design rationale:
    - Uses HuggingFace tokenizer to ensure accurate token counting
    - Respects chunk_size limit while maintaining semantic boundaries
    - Configurable overlap to prevent information loss at boundaries
    """
    
    def __init__(self, config: ChunkingConfig, tokenizer_model: str):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        logger.info(f"Initialized chunker with config: {config}")
    
    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """
        Convert documents to chunks.
        
        Args:
            documents: List of Document objects
        
        Returns:
            List of Chunk objects
        """
        chunks = []
        with timer(f"Chunking {len(documents)} documents"):
            for doc in documents:
                doc_chunks = self._chunk_document(doc)
                chunks.extend(doc_chunks)
                logger.info(
                    f"Document {doc.doc_id}: created {len(doc_chunks)} chunks "
                    f"(avg {self.config.chunk_size} tokens)"
                )
        
        return chunks
    
    def _chunk_document(self, document: Document) -> List[Chunk]:
        """
        Chunk a single document into Chunk objects.
        
        Args:
            document: Document to chunk
        
        Returns:
            List of chunks from the document
        """
        from langchain_text_splitters import CharacterTextSplitter
        
        # Use character-based splitter with exact token counting
        text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.tokenizer,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        
        split_texts = text_splitter.split_text(document.content)
        
        chunks = []
        for chunk_index, chunk_text in enumerate(split_texts):
            # Skip very small chunks
            if len(chunk_text) < self.config.min_chunk_length:
                logger.debug(f"Skipping chunk {chunk_index} (too small)")
                continue
            
            # Count actual tokens in this chunk
            token_count = len(self.tokenizer.encode(chunk_text))
            
            chunk = Chunk(
                content=chunk_text,
                chunk_id=self._generate_chunk_id(document.doc_id, chunk_index),
                doc_id=document.doc_id,
                chunk_index=chunk_index,
                tokens=token_count,
                source=document.source,
                metadata={
                    **document.metadata,
                    'chunk_index': chunk_index,
                    'total_tokens': token_count,
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    @staticmethod
    def _generate_chunk_id(doc_id: str, chunk_index: int) -> str:
        """Generate deterministic chunk ID."""
        combined = f"{doc_id}_{chunk_index}"
        return hashlib.md5(combined.encode()).hexdigest()[:12]


class IngestionPipeline:
    """
    Complete ingestion pipeline orchestrating loading and chunking.
    
    This is the main entry point for document ingestion.
    """
    
    def __init__(self, config: ChunkingConfig, tokenizer_model: str):
        self.config = config
        self.chunker = DocumentChunker(config, tokenizer_model)
        logger.info("Initialized ingestion pipeline")
    
    def ingest(self, file_path: str) -> List[Chunk]:
        """
        End-to-end ingestion: load and chunk a document.
        
        Args:
            file_path: Path to text document
        
        Returns:
            List of chunks ready for embedding
        """
        logger.info(f"Starting ingestion pipeline for {file_path}")
        
        # Load documents
        loader = TextFileLoader(file_path)
        documents = loader.load()
        
        # Chunk documents
        chunks = self.chunker.chunk_documents(documents)
        
        logger.info(
            f"Ingestion complete: {len(documents)} document(s), "
            f"{len(chunks)} chunk(s)"
        )
        
        return chunks
