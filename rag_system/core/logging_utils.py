"""
Logging and utility functions for the RAG system.
"""
import logging
import time
from functools import wraps
from typing import Callable, Any
from contextlib import contextmanager


def setup_logging(level=logging.INFO, log_file: str = None):
    """
    Configure logging for the RAG system.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path to write logs
    """
    logger = logging.getLogger("rag_system")
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module."""
    return logging.getLogger(f"rag_system.{name}")


@contextmanager
def timer(description: str = "Operation"):
    """
    Context manager for timing operations.
    
    Usage:
        with timer("Embedding documents"):
            # do work
    """
    logger = get_logger("timer")
    start = time.time()
    logger.debug(f"[START] {description}")
    try:
        yield
    finally:
        elapsed_ms = (time.time() - start) * 1000
        logger.debug(f"[END] {description} - {elapsed_ms:.2f}ms")


def timeit(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Usage:
        @timeit
        def my_function():
            pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger("timeit")
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_ms = (time.time() - start) * 1000
        logger.debug(f"{func.__name__} took {elapsed_ms:.2f}ms")
        return result
    return wrapper


# Initialize default logger
setup_logging()
