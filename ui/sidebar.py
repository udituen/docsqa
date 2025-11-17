"""Sidebar UI components."""

import streamlit as st
from config import SAMPLE_TEXT, EXAMPLE_QUESTIONS
from utils.document_processor import read_uploaded_file
from models.retriever import build_retriever
from models.llm_loader import load_qwen_llm
from chains.qa_chain import create_qa_chain
from config import QWEN_MODEL_NAME, EMBEDDING_MODEL_NAME, MAX_NEW_TOKENS, TEMPERATURE, TOP_P


def render_sidebar():
    """Render the sidebar with upload and controls."""
    with st.sidebar:
        st.header("📄 Document Upload")
        
        # Sample file download
        st.download_button(
            label="📄 Download Sample File",
            data=SAMPLE_TEXT,
            file_name="sample_agri.txt",
            mime="text/plain"
        )
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload your file", 
            type=["txt", "pdf"]
        )
        
        if uploaded_file is not None:
            st.success(f"{uploaded_file.name}")
            _handle_document_upload(uploaded_file)
        
        # Example questions
        if st.session_state.document_processed:
            _render_example_questions()
        
        # Clear chat button
        if st.session_state.chat_history:
            _render_clear_button()


def _handle_document_upload(uploaded_file):
    """Handle document processing."""
    if st.button("Process Document", type="primary"):
        with st.spinner("Processing document..."):
            try:
                docs = read_uploaded_file(uploaded_file)
                
                if len(docs) > 0:
                    retriever = build_retriever(docs, EMBEDDING_MODEL_NAME)
                    llm = load_qwen_llm(
                        QWEN_MODEL_NAME,
                        MAX_NEW_TOKENS,
                        TEMPERATURE,
                        TOP_P
                    )
                    
                    st.session_state.qa_chain = create_qa_chain(llm, retriever)
                    st.session_state.document_processed = True
                    st.session_state.chat_history = []
                    
                    st.success(f"Processed {len(docs)} text chunks!")
                    st.rerun()
                else:
                    st.error("No content found in file.")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")


def _render_example_questions():
    """Render example question buttons."""
    st.markdown("---")
    st.subheader("💡 Example Questions")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, key=f"example_{q}"):
            st.session_state.user_input = q
            st.rerun()


def _render_clear_button():
    """Render clear chat history button."""
    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
