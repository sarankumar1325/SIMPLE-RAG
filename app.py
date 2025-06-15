"""
Streamlit web application for the RAG system.
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
import time
from typing import List, Dict, Any

from rag_pipeline import RAGPipeline
from config import Config
from utils import safe_filename, validate_file_type, validate_file_size

# Page configuration
st.set_page_config(
    page_title="Simple RAG Application",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .status-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .status-info {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    .source-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_pipeline():
    """Initialize the RAG pipeline (cached for performance)."""
    try:
        config = Config()
        config.validate_config()
        return RAGPipeline()
    except Exception as e:
        st.error(f"Failed to initialize RAG pipeline: {e}")
        return None

def display_status_box(message: str, status_type: str = "info"):
    """Display a styled status box."""
    css_class = f"status-{status_type}"
    st.markdown(f'<div class="status-box {css_class}">{message}</div>', unsafe_allow_html=True)

def handle_file_upload(uploaded_files: List, rag_pipeline: RAGPipeline) -> Dict[str, Any]:
    """Handle file upload and processing."""
    if not uploaded_files:
        return {"success": False, "message": "No files uploaded"}
    
    config = Config()
    temp_files = []
    
    try:
        # Save uploaded files to temporary directory
        for uploaded_file in uploaded_files:
            # Validate file type
            if not validate_file_type(uploaded_file.name, config.SUPPORTED_FILE_TYPES):
                st.error(f"Unsupported file type: {uploaded_file.name}")
                continue
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_files.append(tmp_file.name)
        
        if not temp_files:
            return {"success": False, "message": "No valid files to process"}
        
        # Process files
        if len(temp_files) == 1:
            result = rag_pipeline.ingest_document(temp_files[0])
        else:
            result = rag_pipeline.ingest_multiple_documents(temp_files)
        
        return result
        
    except Exception as e:
        return {"success": False, "error": str(e)}
    
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass

def display_sources(sources: List[Dict[str, Any]]):
    """Display source information in a formatted way."""
    if not sources:
        return
    
    st.subheader("📚 Sources")
    
    for source in sources:
        with st.expander(f"Source {source.get('id', '?')}: {source.get('filename', 'Unknown')} (Score: {source.get('score', 'N/A')})"):
            if 'chunk' in source:
                st.write(f"**Chunk:** {source['chunk']}")
            
            if 'content_preview' in source:
                st.write("**Content Preview:**")
                st.write(source['content_preview'])

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Simple RAG Application</h1>', unsafe_allow_html=True)
    st.markdown("**Retrieval-Augmented Generation with ChromaDB and Gemini API**")
    
    # Initialize RAG pipeline
    rag_pipeline = initialize_rag_pipeline()
    
    if not rag_pipeline:
        st.error("Failed to initialize the application. Please check your configuration.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # System status
        with st.expander("System Status", expanded=False):
            if st.button("Check Status"):
                with st.spinner("Checking system status..."):
                    status = rag_pipeline.get_system_status()
                    
                    if "error" in status:
                        st.error(f"System Error: {status['error']}")
                    else:
                        # Vector store status
                        vs_status = status.get("vector_store", {})
                        st.write(f"**Vector Store:** {vs_status.get('status', 'unknown').title()}")
                        st.write(f"**Documents:** {vs_status.get('document_count', 0)}")
                        st.write(f"**Embedding Model:** {vs_status.get('embedding_model', 'unknown')}")
                        
                        # Gemini API status
                        gemini_status = status.get("gemini_api", {})
                        st.write(f"**Gemini API:** {gemini_status.get('status', 'unknown').title()}")
        
        # Reset system
        with st.expander("Reset System", expanded=False):
            st.warning("This will delete all uploaded documents!")
            if st.button("Reset All Documents", type="secondary"):
                with st.spinner("Resetting system..."):
                    result = rag_pipeline.reset_system()
                    if result["success"]:
                        st.success(result["message"])
                        st.rerun()
                    else:
                        st.error(f"Reset failed: {result.get('error', 'Unknown error')}")
        
        # Configuration display
        with st.expander("Current Settings", expanded=False):
            config = Config()
            st.write(f"**Chunk Size:** {config.CHUNK_SIZE}")
            st.write(f"**Chunk Overlap:** {config.CHUNK_OVERLAP}")
            st.write(f"**Max Retrieval Results:** {config.MAX_RETRIEVAL_RESULTS}")
            st.write(f"**Temperature:** {config.TEMPERATURE}")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 Upload Documents", "🔍 Search"])
    
    # Chat Tab
    with tab1:
        st.header("Ask Questions About Your Documents")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Display sources if available
                if message["role"] == "assistant" and "sources" in message:
                    display_sources(message["sources"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = rag_pipeline.query(prompt)
                    
                    # Display answer
                    st.markdown(response["answer"])
                    
                    # Display sources
                    if response.get("sources"):
                        display_sources(response["sources"])
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response.get("sources", [])
                    })
    
    # Upload Tab
    with tab2:
        st.header("Upload Documents")
        st.write("Upload PDF, TXT, or DOCX files to add them to your knowledge base.")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, DOCX"
        )
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} file(s):")
            for file in uploaded_files:
                st.write(f"- {file.name} ({file.size} bytes)")
            
            if st.button("Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    result = handle_file_upload(uploaded_files, rag_pipeline)
                    
                    if result["success"]:
                        if "chunks_created" in result:
                            # Single document
                            display_status_box(result["message"], "success")
                        else:
                            # Multiple documents
                            success_msg = f"Successfully processed {result['successful_ingestions']}/{result['total_documents']} documents"
                            success_msg += f" ({result['total_chunks_created']} chunks created)"
                            display_status_box(success_msg, "success")
                            
                            # Show individual results
                            with st.expander("Detailed Results"):
                                for individual_result in result.get("individual_results", []):
                                    file_name = Path(individual_result["file_path"]).name
                                    if individual_result["success"]:
                                        st.success(f"✅ {file_name}: {individual_result['chunks_created']} chunks")
                                    else:
                                        st.error(f"❌ {file_name}: {individual_result.get('error', 'Unknown error')}")
                    else:
                        display_status_box(f"Error: {result.get('error', 'Unknown error')}", "error")
    
    # Search Tab
    with tab3:
        st.header("Search Documents")
        st.write("Search through your uploaded documents without generating an AI response.")
        
        search_query = st.text_input("Enter search query:", placeholder="Search for specific content...")
        
        if search_query and st.button("Search", type="primary"):
            with st.spinner("Searching..."):
                results = rag_pipeline.search_documents(search_query)
                
                if results:
                    st.success(f"Found {len(results)} relevant documents")
                    
                    for i, result in enumerate(results, 1):
                        with st.expander(f"Result {i} - Score: {result.get('score', 0):.3f}"):
                            metadata = result.get("metadata", {})
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**File:** {metadata.get('filename', 'Unknown')}")
                                st.write(f"**Type:** {metadata.get('file_type', 'Unknown')}")
                            
                            with col2:
                                if 'chunk_id' in metadata:
                                    st.write(f"**Chunk:** {metadata['chunk_id'] + 1}/{metadata.get('chunk_count', '?')}")
                                st.write(f"**Score:** {result.get('score', 0):.3f}")
                            
                            st.write("**Content:**")
                            st.write(result.get("content", "No content available"))
                else:
                    st.info("No relevant documents found for your search query.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using [Streamlit](https://streamlit.io/), "
        "[ChromaDB](https://www.trychroma.com/), and "
        "[Google Gemini API](https://ai.google.dev/)"
    )

if __name__ == "__main__":
    main()

