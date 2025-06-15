"""
Vector store module using ChromaDB for the RAG application.
Handles embedding generation, storage, and retrieval.
"""

import logging
import uuid
from typing import List, Dict, Optional, Union, Any
import chromadb
from chromadb.config import Settings
import numpy as np

# Simple embedding function using basic text features
class SimpleEmbedding:
    def encode(self, text):
        # Simple bag-of-words style embedding for demo
        words = text.lower().split()
        # Create a simple 384-dimensional vector based on text features
        vector = np.zeros(384)
        for i, word in enumerate(words[:100]):  # Use first 100 words
            hash_val = hash(word) % 384
            vector[hash_val] += 1
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector
from config import Config
from utils import format_retrieved_context

logger = logging.getLogger(__name__)

class VectorStore:
    """ChromaDB-based vector store for document embeddings."""
    
    def __init__(self, collection_name: str = "rag_documents"):
        self.config = Config()
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.config.CHROMA_PERSIST_DIRECTORY,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding model
        self.embedding_model = SimpleEmbedding()
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
        logger.info(f"VectorStore initialized with collection: {collection_name}")
    
    def _get_or_create_collection(self):
        """Get existing collection or create a new one."""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
        except Exception:
            # Create new collection if it doesn't exist
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "RAG document embeddings"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
        
        return collection
    
    def add_documents(self, documents: List[Dict[str, Union[str, Dict]]]) -> bool:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document dictionaries with content and metadata
            
        Returns:
            True if successful, False otherwise
        """
        if not documents:
            logger.warning("No documents to add")
            return False
        
        try:
            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            metadatas = []
            documents_content = []
            
            for doc in documents:
                content = doc["content"]
                metadata = doc["metadata"]
                
                # Generate unique ID
                doc_id = str(uuid.uuid4())
                
                # Generate embedding
                embedding = self.embedding_model.encode(content).tolist()
                
                # Prepare metadata (ChromaDB requires string values)
                clean_metadata = self._clean_metadata(metadata)
                
                ids.append(doc_id)
                embeddings.append(embedding)
                metadatas.append(clean_metadata)
                documents_content.append(content)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents_content
            )
            
            logger.info(f"Successfully added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            return False
    
    def similarity_search(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """
        Perform similarity search for a query.
        
        Args:
            query: Search query string
            k: Number of results to return (default from config)
            
        Returns:
            List of similar documents with metadata and scores
        """
        if k is None:
            k = self.config.MAX_RETRIEVAL_RESULTS
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    formatted_results.append({
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": 1 - results["distances"][0][i],  # Convert distance to similarity
                        "distance": results["distances"][0][i]
                    })
            
            logger.info(f"Retrieved {len(formatted_results)} documents for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            return {
                "document_count": count,
                "collection_name": self.collection_name,
                "embedding_model": self.config.EMBEDDING_MODEL
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def delete_collection(self) -> bool:
        """Delete the entire collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False
    
    def reset_collection(self) -> bool:
        """Reset the collection (delete all documents)."""
        try:
            # Delete and recreate collection
            self.delete_collection()
            self.collection = self._get_or_create_collection()
            logger.info(f"Reset collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            return False
    
    def search_by_metadata(self, metadata_filter: Dict[str, str], k: int = None) -> List[Dict[str, Any]]:
        """
        Search documents by metadata filters.
        
        Args:
            metadata_filter: Dictionary of metadata key-value pairs to filter by
            k: Number of results to return
            
        Returns:
            List of matching documents
        """
        if k is None:
            k = self.config.MAX_RETRIEVAL_RESULTS
        
        try:
            # Build where clause for ChromaDB
            where_clause = {}
            for key, value in metadata_filter.items():
                where_clause[key] = {"$eq": value}
            
            results = self.collection.get(
                where=where_clause,
                limit=k,
                include=["documents", "metadatas"]
            )
            
            # Format results
            formatted_results = []
            if results["documents"]:
                for i in range(len(results["documents"])):
                    formatted_results.append({
                        "content": results["documents"][i],
                        "metadata": results["metadatas"][i],
                        "score": 1.0  # No similarity score for metadata search
                    })
            
            logger.info(f"Found {len(formatted_results)} documents matching metadata filter")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching by metadata: {e}")
            return []
    
    def _clean_metadata(self, metadata: Dict) -> Dict[str, str]:
        """
        Clean metadata for ChromaDB storage (convert all values to strings).
        
        Args:
            metadata: Original metadata dictionary
            
        Returns:
            Cleaned metadata with string values
        """
        clean_metadata = {}
        
        for key, value in metadata.items():
            # Convert all values to strings
            if isinstance(value, (int, float)):
                clean_metadata[key] = str(value)
            elif isinstance(value, bool):
                clean_metadata[key] = str(value).lower()
            elif value is None:
                clean_metadata[key] = "none"
            else:
                clean_metadata[key] = str(value)
        
        return clean_metadata
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding model."""
        try:
            # Generate a test embedding to get dimension
            test_embedding = self.embedding_model.encode("test")
            return len(test_embedding)
        except Exception as e:
            logger.error(f"Error getting embedding dimension: {e}")
            return 384  # Default for all-MiniLM-L6-v2
    
    def batch_add_documents(self, documents: List[Dict[str, Union[str, Dict]]], batch_size: int = 100) -> bool:
        """
        Add documents in batches for better performance with large datasets.
        
        Args:
            documents: List of document dictionaries
            batch_size: Number of documents to process in each batch
            
        Returns:
            True if all batches successful, False otherwise
        """
        if not documents:
            return False
        
        total_batches = (len(documents) + batch_size - 1) // batch_size
        successful_batches = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
            
            if self.add_documents(batch):
                successful_batches += 1
            else:
                logger.error(f"Failed to process batch {batch_num}")
        
        success_rate = successful_batches / total_batches
        logger.info(f"Batch processing complete: {successful_batches}/{total_batches} batches successful")
        
        return success_rate == 1.0
