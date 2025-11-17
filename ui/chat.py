"""Chat interface components."""

import streamlit as st


def render_chat_interface():
    """Render the main chat interface."""
    if not st.session_state.document_processed:
        st.info("<-- Please upload a document in the sidebar and click 'Process Document' to start chatting!")
        return
    
    # Display chat history
    _display_chat_history()
    
    # Handle new user input
    _handle_user_input()


def _display_chat_history():
    """Display all messages in chat history."""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("View Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**Source {i+1}:** {source}")


def _handle_user_input():
    """Handle new user input and generate response."""
    if prompt := st.chat_input("Ask a question about your document..."):
        # Add user message
        st.session_state.chat_history.append({
            "role": "user", 
            "content": prompt
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display response
        _generate_response(prompt)


def _generate_response(prompt):
    """Generate AI response to user prompt."""
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.qa_chain({"question": prompt})
                
                answer = result["answer"]
                sources = [
                    doc.page_content 
                    for doc in result.get("source_documents", [])
                ]
                
                st.markdown(answer)
                
                # Show sources
                if sources:
                    with st.expander("View Sources"):
                        for i, source in enumerate(sources):
                            st.markdown(f"**Source {i+1}:** {source}")
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_msg
                })
