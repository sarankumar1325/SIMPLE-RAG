"""
Main RAG pipeline orchestrating document processing, vector storage, and generation.
"""

import logging
from typing import List, Dict, Optional, Union, Any
from pathlib import Path
import time

from document_processor import DocumentProcessor
from vector_store import VectorStore
from gemini_client import GeminiClient
from config import Config
from utils import setup_logging

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Main RAG pipeline orchestrating all components."""
    
    def __init__(self, collection_name: str = "rag_documents"):
        """
        Initialize the RAG pipeline.
        
        Args:
            collection_name: Name for the vector store collection
        """
        self.config = Config()
        
        # Validate configuration
        self.config.validate_config()
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore(collection_name=collection_name)
        self.gemini_client = GeminiClient()
        
        logger.info("RAG Pipeline initialized successfully")
    
    def ingest_document(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Ingest a single document into the RAG system.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with ingestion results and metadata
        """
        start_time = time.time()
        file_path = Path(file_path)
        
        try:
            logger.info(f"Starting document ingestion: {file_path}")
            
            # Process the document
            chunks = self.document_processor.process_document(file_path)
            
            if not chunks:
                return {
                    "success": False,
                    "error": "No content extracted from document",
                    "file_path": str(file_path)
                }
            
            # Add chunks to vector store
            success = self.vector_store.add_documents(chunks)
            
            processing_time = time.time() - start_time
            
            if success:
                logger.info(f"Successfully ingested document: {file_path} ({len(chunks)} chunks)")
                return {
                    "success": True,
                    "file_path": str(file_path),
                    "chunks_created": len(chunks),
                    "processing_time": processing_time,
                    "message": f"Successfully processed {file_path.name} into {len(chunks)} chunks"
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to add document to vector store",
                    "file_path": str(file_path)
                }
                
        except Exception as e:
            logger.error(f"Error ingesting document {file_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_path": str(file_path)
            }
    
    def ingest_multiple_documents(self, file_paths: List[Union[str, Path]]) -> Dict[str, Any]:
        """
        Ingest multiple documents into the RAG system.
        
        Args:
            file_paths: List of paths to document files
            
        Returns:
            Dictionary with batch ingestion results
        """
        start_time = time.time()
        results = []
        successful_ingestions = 0
        total_chunks = 0
        
        logger.info(f"Starting batch ingestion of {len(file_paths)} documents")
        
        for file_path in file_paths:
            result = self.ingest_document(file_path)
            results.append(result)
            
            if result["success"]:
                successful_ingestions += 1
                total_chunks += result.get("chunks_created", 0)
        
        processing_time = time.time() - start_time
        
        return {
            "success": successful_ingestions > 0,
            "total_documents": len(file_paths),
            "successful_ingestions": successful_ingestions,
            "failed_ingestions": len(file_paths) - successful_ingestions,
            "total_chunks_created": total_chunks,
            "processing_time": processing_time,
            "individual_results": results
        }
    
    def query(self, question: str, include_sources: bool = True) -> Dict[str, Any]:
        """
        Query the RAG system with a question.
        
        Args:
            question: User's question
            include_sources: Whether to include source information in response
            
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing query: {question[:100]}...")
            
            # Check if vector store has any documents
            stats = self.vector_store.get_collection_stats()
            if stats.get("document_count", 0) == 0:
                return {
                    "answer": "I don't have any documents to search through. Please upload some documents first.",
                    "success": False,
                    "error": "No documents in vector store",
                    "sources": []
                }
            
            # Retrieve relevant documents
            retrieved_docs = self.vector_store.similarity_search(
                query=question,
                k=self.config.MAX_RETRIEVAL_RESULTS
            )
            
            if not retrieved_docs:
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "success": False,
                    "error": "No relevant documents found",
                    "sources": []
                }
            
            # Generate response using Gemini
            response = self.gemini_client.generate_rag_response(question, retrieved_docs)
            
            # Add timing and source information
            response["query_time"] = time.time() - start_time
            response["retrieved_chunks"] = len(retrieved_docs)
            
            if include_sources and "sources" not in response:
                response["sources"] = self._format_sources(retrieved_docs)
            
            logger.info(f"Query processed successfully in {response['query_time']:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "success": False,
                "error": str(e),
                "sources": []
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get the current status of the RAG system.
        
        Returns:
            Dictionary with system status information
        """
        try:
            # Vector store stats
            vector_stats = self.vector_store.get_collection_stats()
            
            # Test Gemini connection
            gemini_status = self.gemini_client.test_connection()
            
            return {
                "vector_store": {
                    "status": "healthy" if vector_stats.get("document_count", 0) >= 0 else "error",
                    "document_count": vector_stats.get("document_count", 0),
                    "collection_name": vector_stats.get("collection_name", "unknown"),
                    "embedding_model": vector_stats.get("embedding_model", "unknown")
                },
                "gemini_api": {
                    "status": "healthy" if gemini_status else "error",
                    "connection_test": gemini_status
                },
                "configuration": {
                    "chunk_size": self.config.CHUNK_SIZE,
                    "chunk_overlap": self.config.CHUNK_OVERLAP,
                    "max_retrieval_results": self.config.MAX_RETRIEVAL_RESULTS,
                    "temperature": self.config.TEMPERATURE
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    def reset_system(self) -> Dict[str, Any]:
        """
        Reset the RAG system (clear all documents).
        
        Returns:
            Dictionary with reset results
        """
        try:
            logger.info("Resetting RAG system...")
            
            success = self.vector_store.reset_collection()
            
            if success:
                logger.info("RAG system reset successfully")
                return {
                    "success": True,
                    "message": "RAG system has been reset. All documents have been removed."
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to reset vector store"
                }
                
        except Exception as e:
            logger.error(f"Error resetting system: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def search_documents(self, query: str, search_type: str = "similarity") -> List[Dict[str, Any]]:
        """
        Search for documents without generating a response.
        
        Args:
            query: Search query
            search_type: Type of search ("similarity" or "metadata")
            
        Returns:
            List of matching documents
        """
        try:
            if search_type == "similarity":
                return self.vector_store.similarity_search(query)
            else:
                # For metadata search, parse query as key:value pairs
                # This is a simple implementation - could be enhanced
                return []
                
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def get_document_summary(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get a summary of a document without ingesting it.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Dictionary with document summary
        """
        try:
            # Load document content
            document = self.document_processor.load_document(file_path)
            
            # Generate summary using Gemini
            summary_result = self.gemini_client.summarize_document(document["content"])
            
            # Add document metadata
            summary_result["metadata"] = document["metadata"]
            
            return summary_result
            
        except Exception as e:
            logger.error(f"Error generating document summary: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _format_sources(self, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Format source information for display."""
        sources = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            metadata = doc.get("metadata", {})
            source = {
                "id": i,
                "filename": metadata.get("filename", "Unknown"),
                "score": f"{doc.get('score', 0):.3f}",
                "content_preview": doc.get("content", "")[:200] + "..." if len(doc.get("content", "")) > 200 else doc.get("content", "")
            }
            
            if "chunk_id" in metadata:
                source["chunk"] = f"{metadata['chunk_id'] + 1}/{metadata.get('chunk_count', '?')}"
            
            sources.append(source)
        
        return sources

