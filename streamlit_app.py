"""Main Streamlit application."""

import streamlit as st
from ui.sidebar import render_sidebar
from ui.chat import render_chat_interface


def initialize_session_state():
    """Initialize session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'document_processed' not in st.session_state:
        st.session_state.document_processed = False


def main():
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title="DocsQA",
        page_icon="",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("DocsQA: Chat with Your Document")
    st.markdown("Upload a document and have a conversation about its contents! (Powered by Qwen)")
    
    # Render UI components
    render_sidebar()
    render_chat_interface()


if __name__ == "__main__":
    main()