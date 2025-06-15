"""
Utility functions for the RAG application.
"""

import os
import re
import logging
import hashlib
from typing import List, Optional, Union
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('rag_app.log')
        ]
    )
    return logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:()\-\'""]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def get_file_hash(file_path: Union[str, Path]) -> str:
    """Generate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Error generating hash for {file_path}: {e}")
        return ""

def validate_file_type(file_path: Union[str, Path], supported_types: List[str]) -> bool:
    """Validate if file type is supported."""
    file_extension = Path(file_path).suffix.lower()
    return file_extension in supported_types

def validate_file_size(file_path: Union[str, Path], max_size_mb: int) -> bool:
    """Validate if file size is within limits."""
    try:
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        return file_size_mb <= max_size_mb
    except Exception as e:
        logger.error(f"Error checking file size for {file_path}: {e}")
        return False

def safe_filename(filename: str) -> str:
    """Generate a safe filename by removing/replacing problematic characters."""
    # Remove or replace problematic characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    
    # Ensure filename is not empty
    if not filename:
        filename = "unnamed_file"
    
    return filename

def chunk_text_by_sentences(text: str, max_chunk_size: int, overlap: int = 0) -> List[str]:
    """
    Chunk text by sentences while respecting max_chunk_size.
    This is a simple sentence-aware chunking strategy.
    """
    if not text:
        return []
    
    # Split by sentences (simple approach)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed max_chunk_size
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            
            # Handle overlap
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + " " + sentence
            else:
                current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    # Add the last chunk if it exists
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def format_retrieved_context(retrieved_docs: List[dict]) -> str:
    """Format retrieved documents into a context string."""
    if not retrieved_docs:
        return ""
    
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        content = doc.get('content', doc.get('page_content', ''))
        metadata = doc.get('metadata', {})
        source = metadata.get('source', 'Unknown')
        
        context_parts.append(f"[Source {i}: {source}]\n{content}")
    
    return "\n\n".join(context_parts)

def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to max_length with optional suffix."""
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract simple keywords from text (basic implementation)."""
    if not text:
        return []
    
    # Simple keyword extraction - remove common stop words and get unique words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
    }
    
    # Extract words and filter
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    keywords = [word for word in words if word not in stop_words]
    
    # Get unique keywords and limit count
    unique_keywords = list(dict.fromkeys(keywords))  # Preserve order while removing duplicates
    
    return unique_keywords[:max_keywords]

