"""
Configuration management for the RAG application.
"""

import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the RAG application."""
    
    # API Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # Vector Database Configuration
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Text Processing Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Retrieval Configuration
    MAX_RETRIEVAL_RESULTS: int = int(os.getenv("MAX_RETRIEVAL_RESULTS", "5"))
    
    # Generation Configuration
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_OUTPUT_TOKENS: int = int(os.getenv("MAX_OUTPUT_TOKENS", "1024"))
    
    # Application Configuration
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")
    SUPPORTED_FILE_TYPES: list = [".pdf", ".txt", ".docx"]
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that all required configuration is present."""
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required. Please set it in your .env file.")
        
        # Create necessary directories
        os.makedirs(cls.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        os.makedirs(cls.UPLOAD_DIR, exist_ok=True)
        
        return True
    
    @classmethod
    def get_gemini_config(cls) -> dict:
        """Get Gemini API configuration."""
        return {
            "api_key": cls.GEMINI_API_KEY,
            "temperature": cls.TEMPERATURE,
            "max_output_tokens": cls.MAX_OUTPUT_TOKENS
        }
    
    @classmethod
    def get_chunking_config(cls) -> dict:
        """Get text chunking configuration."""
        return {
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP
        }
    
    @classmethod
    def get_retrieval_config(cls) -> dict:
        """Get retrieval configuration."""
        return {
            "max_results": cls.MAX_RETRIEVAL_RESULTS,
            "embedding_model": cls.EMBEDDING_MODEL
        }

