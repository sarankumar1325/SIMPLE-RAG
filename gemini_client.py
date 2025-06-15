"""
Gemini API client for the RAG application.
Handles text generation using Google's Gemini API.
"""

import logging
from typing import List, Dict, Optional, Any
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from config import Config
from utils import format_retrieved_context, truncate_text

logger = logging.getLogger(__name__)

class GeminiClient:
    """Client for Google's Gemini API."""
    
    def __init__(self):
        self.config = Config()
        
        # Configure Gemini API
        genai.configure(api_key=self.config.GEMINI_API_KEY)
        
        # Initialize the model
        self.model = genai.GenerativeModel(
            model_name="gemini-pro",
            generation_config=genai.types.GenerationConfig(
                temperature=self.config.TEMPERATURE,
                max_output_tokens=self.config.MAX_OUTPUT_TOKENS,
                top_p=0.8,
                top_k=40
            ),
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
        )
        
        logger.info("GeminiClient initialized successfully")
    
    def generate_rag_response(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a response using RAG (Retrieval-Augmented Generation).
        
        Args:
            query: User's question/query
            retrieved_docs: List of retrieved documents from vector store
            
        Returns:
            Dictionary containing the response and metadata
        """
        try:
            # Format the context from retrieved documents
            context = format_retrieved_context(retrieved_docs)
            
            # Create the RAG prompt
            prompt = self._create_rag_prompt(query, context)
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            # Process the response
            if response.text:
                return {
                    "answer": response.text.strip(),
                    "sources_used": len(retrieved_docs),
                    "context_length": len(context),
                    "prompt_tokens": len(prompt.split()),
                    "success": True,
                    "sources": self._extract_source_info(retrieved_docs)
                }
            else:
                logger.warning("Empty response from Gemini API")
                return {
                    "answer": "I apologize, but I couldn't generate a response to your question.",
                    "success": False,
                    "error": "Empty response from API"
                }
                
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    def generate_simple_response(self, query: str) -> Dict[str, Any]:
        """
        Generate a simple response without RAG context.
        
        Args:
            query: User's question/query
            
        Returns:
            Dictionary containing the response and metadata
        """
        try:
            response = self.model.generate_content(query)
            
            if response.text:
                return {
                    "answer": response.text.strip(),
                    "success": True,
                    "type": "simple"
                }
            else:
                return {
                    "answer": "I couldn't generate a response to your question.",
                    "success": False,
                    "error": "Empty response from API"
                }
                
        except Exception as e:
            logger.error(f"Error generating simple response: {e}")
            return {
                "answer": f"I encountered an error: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    def _create_rag_prompt(self, query: str, context: str) -> str:
        """
        Create a well-structured prompt for RAG.
        
        Args:
            query: User's question
            context: Retrieved context from documents
            
        Returns:
            Formatted prompt string
        """
        # Truncate context if it's too long
        max_context_length = 8000  # Leave room for query and instructions
        if len(context) > max_context_length:
            context = truncate_text(context, max_context_length, "... [context truncated]")
        
        prompt = f"""You are a helpful AI assistant that answers questions based on the provided context. 

INSTRUCTIONS:
1. Answer the question using ONLY the information provided in the context below
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Be specific and cite relevant parts of the context when possible
4. If you're unsure about something, express that uncertainty
5. Provide a clear, well-structured answer

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
        
        return prompt
    
    def _extract_source_info(self, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Extract source information from retrieved documents.
        
        Args:
            retrieved_docs: List of retrieved documents
            
        Returns:
            List of source information dictionaries
        """
        sources = []
        
        for doc in retrieved_docs:
            metadata = doc.get("metadata", {})
            source_info = {
                "filename": metadata.get("filename", "Unknown"),
                "source": metadata.get("source", "Unknown"),
                "score": f"{doc.get('score', 0):.3f}"
            }
            
            # Add chunk info if available
            if "chunk_id" in metadata:
                source_info["chunk"] = f"{metadata['chunk_id'] + 1}/{metadata.get('chunk_count', '?')}"
            
            sources.append(source_info)
        
        return sources
    
    def summarize_document(self, content: str, max_length: int = 500) -> Dict[str, Any]:
        """
        Generate a summary of document content.
        
        Args:
            content: Document content to summarize
            max_length: Maximum length of summary
            
        Returns:
            Dictionary containing summary and metadata
        """
        try:
            # Truncate content if too long
            max_content_length = 10000
            if len(content) > max_content_length:
                content = truncate_text(content, max_content_length)
            
            prompt = f"""Please provide a concise summary of the following text in approximately {max_length} characters:

{content}

Summary:"""
            
            response = self.model.generate_content(prompt)
            
            if response.text:
                summary = response.text.strip()
                return {
                    "summary": summary,
                    "original_length": len(content),
                    "summary_length": len(summary),
                    "success": True
                }
            else:
                return {
                    "summary": "Could not generate summary",
                    "success": False,
                    "error": "Empty response from API"
                }
                
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {
                "summary": f"Error generating summary: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    def generate_keywords(self, content: str, max_keywords: int = 10) -> List[str]:
        """
        Generate keywords from content using Gemini.
        
        Args:
            content: Text content to extract keywords from
            max_keywords: Maximum number of keywords to generate
            
        Returns:
            List of keywords
        """
        try:
            # Truncate content if too long
            if len(content) > 5000:
                content = truncate_text(content, 5000)
            
            prompt = f"""Extract the {max_keywords} most important keywords or key phrases from the following text. 
Return only the keywords, separated by commas:

{content}

Keywords:"""
            
            response = self.model.generate_content(prompt)
            
            if response.text:
                keywords_text = response.text.strip()
                keywords = [kw.strip() for kw in keywords_text.split(",")]
                return keywords[:max_keywords]
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error generating keywords: {e}")
            return []
    
    def test_connection(self) -> bool:
        """
        Test the connection to Gemini API.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = self.model.generate_content("Hello, this is a test.")
            return response.text is not None
        except Exception as e:
            logger.error(f"Gemini API connection test failed: {e}")
            return False

