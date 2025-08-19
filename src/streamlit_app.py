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
    text = uploaded_file.read().decode("utf-8")
    docs = text.split("\n")
    return docs

# Load lightweight LLM
@st.cache_resource
def load_llm():
    pipe = pipeline("text-generation", model="google/flan-t5-small", max_new_tokens=256)
    return HuggingFacePipeline(pipeline=pipe)

# extract 


# Build retriever from uploaded content
def build_retriever(docs):
    # if docs.type == pdf
    # use langchain pymupdf to extract the text from the document

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

uploaded_file = st.file_uploader("Upload your file", type=["txt","pdf"])
if uploaded_file is not None:
    st.write("Filename:", uploaded_file.name)
    data = uploaded_file.read()

    if uploaded_file.type == "text/plain":
        st.text_area("Content", data.decode("utf-8"), height=300)
    else:
        st.info(f"Uploaded {len(data)} bytes (PDF or other format)")
query = st.text_input("Ask a question ")

if uploaded_file is not None:
    st.success("file uploaded")
    docs = read_uploaded_file(uploaded_file)
    retriever = build_retriever(docs)
    llm = load_llm()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    if query:
        with st.spinner("Generating answer..."):
            result = qa_chain.run(query)
        st.success(result)
else:
    st.info("Please upload a `.txt` file or use the sample provided.")
