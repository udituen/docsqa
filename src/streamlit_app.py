import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from transformers import pipeline
import io

# For PDF processing
try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

# ----------------------
# Sample Text Content
# ----------------------
SAMPLE_TEXT = """Fertilizers help improve soil nutrients and crop yield.
Irrigation methods vary depending on climate and crop type.
Crop rotation can enhance soil health and reduce pests.
Composting is an organic way to enrich the soil.
Weed management is essential for higher productivity."""

EXAMPLE_QUESTIONS = [
    "What is this document about?",
    "What is the role of fertilizers in agriculture?",
    "Why is crop rotation important?",
    "How does composting help farming?",
]

# Helper: Read uploaded file (TXT or PDF)
def read_uploaded_file(uploaded_file):
    uploaded_file.seek(0)
    
    if uploaded_file.type == "application/pdf":
        # Handle PDF files
        pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    else:
        # Handle text files
        text = uploaded_file.read().decode("utf-8")
    
    # Split into chunks by lines
    docs = text.split("\n")
    docs = [doc.strip() for doc in docs if doc.strip()]
    return docs

# Load lightweight LLM
@st.cache_resource
def load_llm():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_length=256,
        temperature=0.7,
        top_p=0.95
    )
    return HuggingFacePipeline(pipeline=pipe)

# Build retriever from uploaded content
def build_retriever(docs):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_texts(docs, embeddings)
    return db.as_retriever()

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False

# Streamlit UI
st.title("DocsQA: Chat with Your Document💬")

st.markdown("Upload a document and have a conversation about its contents!")

# Sidebar for document upload
with st.sidebar:
    st.header("📄 Document Upload")
    
    # Add sample file download button
    st.download_button(
        label="📥 Download Sample File",
        data=SAMPLE_TEXT,
        file_name="sample_agri.txt",
        mime="text/plain"
    )
    
    uploaded_file = st.file_uploader("Upload your file", type=["txt", "pdf"])
    
    if uploaded_file is not None:
        st.success(f"{uploaded_file.name}")
        
        # Process document button
        if st.button("Process Document", type="primary"):
            with st.spinner("Processing document..."):
                try:
                    docs = read_uploaded_file(uploaded_file)
                    
                    if len(docs) > 0:
                        retriever = build_retriever(docs)
                        llm = load_llm()
                        
                        # Create conversational chain with memory
                        memory = ConversationBufferMemory(
                            memory_key="chat_history",
                            return_messages=True,
                            output_key="answer"
                        )
                        
                        st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
                            llm=llm,
                            retriever=retriever,
                            memory=memory,
                            return_source_documents=True
                        )
                        
                        st.session_state.document_processed = True
                        st.session_state.chat_history = []
                        st.success(f"✅ Processed {len(docs)} text chunks!")
                        st.rerun()
                    else:
                        st.error("No content found in file.")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Show example questions
    if st.session_state.document_processed:
        st.markdown("---")
        st.subheader("💡 Example Questions")
        for q in EXAMPLE_QUESTIONS:
            if st.button(q, key=f"example_{q}"):
                st.session_state.user_input = q
                st.rerun()
    
    # Clear chat button
    if st.session_state.chat_history:
        st.markdown("---")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

# Main chat interface
if not st.session_state.document_processed:
    st.info("👈 Please upload a document in the sidebar and click 'Process Document' to start chatting!")
else:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**Source {i+1}:** {source}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.qa_chain({
                        "question": prompt
                    })
                    
                    answer = result["answer"]
                    sources = [doc.page_content for doc in result.get("source_documents", [])]
                    
                    st.markdown(answer)
                    
                    # Show sources
                    if sources:
                        with st.expander("📚 View Sources"):
                            for i, source in enumerate(sources):
                                st.markdown(f"**Source {i+1}:** {source}")
                    
                    # Add assistant message to chat history
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