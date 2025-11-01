import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

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

# Helper: Read uploaded file
def read_uploaded_file(uploaded_file):
    uploaded_file.seek(0)
    text = uploaded_file.read().decode("utf-8")
    docs = text.split("\n")
    docs = [doc.strip() for doc in docs if doc.strip()]
    return docs

# Load lightweight LLM - FIXED VERSION
@st.cache_resource
def load_llm():
    # Use text2text-generation for FLAN-T5
    pipe = pipeline(
        "text2text-generation",  # ← Changed from text-generation
        model="google/flan-t5-small",
        max_length=256,  # ← Changed from max_new_tokens
        temperature=0.7,
        top_p=0.95
    )
    return HuggingFacePipeline(pipeline=pipe)

# Build retriever from uploaded content
def build_retriever(docs):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_texts(docs, embeddings)
    return db.as_retriever()

# Streamlit UI
st.title("DocsQA: Upload & Ask")

st.markdown("Upload a text file and ask questions about its contents.")

# Add sample file download button
st.download_button(
    label="📄 Download Sample File",
    data=SAMPLE_TEXT,
    file_name="sample_agri.txt",
    mime="text/plain"
)

# Show example questions
with st.expander("💡 Try example questions"):
    for q in EXAMPLE_QUESTIONS:
        st.markdown(f"- {q}")

uploaded_file = st.file_uploader("Upload your file", type=["txt", "pdf"])

if uploaded_file is not None:
    st.write("📁 Filename:", uploaded_file.name)
    
    file_content = uploaded_file.read()
    
    if uploaded_file.type == "text/plain":
        st.text_area("Content Preview", file_content.decode("utf-8"), height=200)
    else:
        st.info(f"Uploaded {len(file_content)} bytes (PDF or other format)")

query = st.text_input("Ask a question")

if uploaded_file is not None:
    docs = read_uploaded_file(uploaded_file)
    
    st.info(f"✅ Extracted {len(docs)} text chunks from document")
    
    if len(docs) > 0:
        retriever = build_retriever(docs)
        llm = load_llm()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            retriever=retriever,
            return_source_documents=True  # Optional: see source docs
        )
        
        if query:
            with st.spinner("Generating answer..."):
                result = qa_chain({"query": query})
                
            st.success("Answer:")
            st.write(result["result"])
            
            # Show source documents
            with st.expander("📄 View source documents"):
                for i, doc in enumerate(result["source_documents"]):
                    st.write(f"**Source {i+1}:** {doc.page_content}")
    else:
        st.error("No content found in file. Please check your file.")
else:
    st.info("Please upload a `.txt` file or use the sample provided.")