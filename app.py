"""
RAG Chatbot - Streamlit Application
A beautiful, modern chat interface for document Q&A with citations
"""

import streamlit as st
import tempfile
import os
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import APP_TITLE, APP_DESCRIPTION, DOCUMENTS_DIR
from src.pipeline import get_pipeline


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="[DOCS]",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS (DISABLED FOR DEBUGGING)
# =============================================================================
# Temporarily disable custom CSS to diagnose blank UI issues. Uncomment the block below
# to restore styling once issue is resolved.
# st.markdown("""
# <style>
#     /* Custom styles removed for debugging */
# </style>
# """, unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pipeline" not in st.session_state:
    st.session_state.pipeline = get_pipeline()

if "documents_ingested" not in st.session_state:
    # Check if there are existing documents in vector store
    stats = st.session_state.pipeline.get_stats()
    st.session_state.documents_ingested = stats["vector_store"]["document_count"] > 0


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown(f"# {APP_TITLE}")
    st.markdown(f"*{APP_DESCRIPTION}*")
    st.markdown("---")
    
    # Stats
    stats = st.session_state.pipeline.get_stats()
    doc_count = stats["vector_store"]["document_count"]
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{doc_count}</div>
            <div>Chunks</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        status = "[OK] Ready" if doc_count > 0 else "[PENDING] Empty"
        st.markdown(f"""
        <div class="stat-card">
            <div style="font-size: 1.5rem;">{'[READY]' if doc_count > 0 else '[WAIT]'}</div>
            <div>{status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### [DIR] Document Ingestion")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "pptx", "ppt", "md", "txt"],
        accept_multiple_files=True,
        help="Supported formats: PDF, PowerPoint, Markdown, Text"
    )
    
    # Ingest options
    col1, col2 = st.columns(2)
    with col1:
        clear_existing = st.checkbox("Clear existing", value=False)
    with col2:
        use_default_dir = st.checkbox("Use data/documents", value=False)
    
    # Ingest button
    if st.button("[RUN] Ingest Documents", use_container_width=True):
        with st.spinner("Processing documents..."):
            try:
                if uploaded_files:
                    # Save uploaded files temporarily
                    temp_paths = []
                    for uploaded_file in uploaded_files:
                        # Create temp file with correct extension
                        suffix = Path(uploaded_file.name).suffix
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(uploaded_file.getvalue())
                            temp_paths.append(tmp.name)
                    
                    # Ingest uploaded files
                    result = st.session_state.pipeline.ingest_uploaded_files(
                        temp_paths,
                        clear_existing=clear_existing
                    )
                    
                    # Clean up temp files
                    for path in temp_paths:
                        try:
                            os.unlink(path)
                        except:
                            pass
                    
                elif use_default_dir:
                    result = st.session_state.pipeline.ingest_documents(
                        str(DOCUMENTS_DIR),
                        clear_existing=clear_existing
                    )
                else:
                    st.warning("Please upload files or check 'Use data/documents'")
                    result = None
                
                if result:
                    if result["success"]:
                        st.success(f"[OK] {result['message']}")
                        st.session_state.documents_ingested = True
                        st.rerun()
                    else:
                        st.error(f"[ERROR] {result['message']}")
                        
            except Exception as e:
                st.error(f"[ERROR] Error: {str(e)}")
    
    # Clear button
    st.markdown("---")
    if st.button("[CLEAR] Clear All Documents", use_container_width=True):
        if st.session_state.pipeline.clear_all():
            st.session_state.documents_ingested = False
            st.session_state.messages = []
            st.success("Cleared all documents")
            st.rerun()
    
    # Clear chat button
    if st.button("[CHAT] Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Help section
    st.markdown("---")
    with st.expander("[INFO] How to use"):
        st.markdown("""
        1. **Upload** your documents (PDF, PPTX, MD, TXT)
        2. Click **Ingest Documents** to process them
        3. **Ask questions** in the chat
        4. Get **answers with citations** from your documents
        
        **Tips:**
        - Be specific in your questions
        - Citations [1], [2] refer to source documents
        - Clear and re-ingest for fresh data
        """)


# =============================================================================
# MAIN CHAT INTERFACE
# =============================================================================
st.markdown(f"# <span class='gradient-text'>{APP_TITLE}</span>", unsafe_allow_html=True)

# Welcome message if no documents
if not st.session_state.documents_ingested:
    st.info("""
    [HELLO] **Welcome!** To get started:
    1. Upload your documents using the sidebar
    2. Click "Ingest Documents" to process them
    3. Start asking questions!
    """)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display citations if present
        if message["role"] == "assistant" and message.get("citations"):
            with st.expander("[DOCS] Sources", expanded=False):
                for cite in message["citations"]:
                    similarity = cite.get('similarity', 0) * 100
                    page_info = f", Page {cite['page']}" if cite.get('page') else ""
                    st.markdown(f"""
                    **[{cite['citation']}]** {cite['source']}{page_info}  
                    <small style='color: #888;'>Relevance: {similarity:.0f}%</small>
                    """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        if not st.session_state.documents_ingested:
            response_text = "[DOCS] Please upload and ingest some documents first using the sidebar."
            st.markdown(response_text)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "citations": []
            })
        else:
            with st.spinner("Thinking..."):
                try:
                    # Query the RAG pipeline
                    result = st.session_state.pipeline.query(prompt)
                    
                    # Display answer
                    st.markdown(result["answer"])
                    
                    # Display citations
                    if result.get("citations"):
                        with st.expander("[DOCS] Sources", expanded=True):
                            for cite in result["citations"]:
                                similarity = cite.get('similarity', 0) * 100
                                page_info = f", Page {cite['page']}" if cite.get('page') else ""
                                st.markdown(f"""
                                **[{cite['citation']}]** {cite['source']}{page_info}  
                                <small style='color: #888;'>Relevance: {similarity:.0f}%</small>
                                """, unsafe_allow_html=True)
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "citations": result.get("citations", [])
                    })
                    
                except Exception as e:
                    error_msg = f"[WARNING] Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "citations": []
                    })


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8rem;'>
    Built with [HEART] using LangChain, ChromaDB, and Streamlit<br>
    <small>RAG Chatbot v1.0</small>
</div>
""", unsafe_allow_html=True)
