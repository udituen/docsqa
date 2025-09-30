import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import create_retrieval_chain
from langchain_community.llms import Ollama
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import itertools
from langchain_community.document_loaders import PyMuPDFLoader

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

HF_TOKEN = st.secrets["HF_TOKEN"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a document question and answer expert.\n"
        "Use the context below to answer the question.\n"
        "Context:\n{context}\n\n"
        "Question: {input}\n"
    )
)

# Helper: Read uploaded file
def read_uploaded_file(uploaded_file):
    return uploaded_file.read().decode("utf-8")

def ingest_pdf(file_name):
    """
    loads content of file using pymupdf
    input (str): file names and file path
    output (list): file content divided by pages
    """
    pages = []
    loader = PyMuPDFLoader(file_name)

    for page in loader.alazy_load():
        pages.append(page)

    return pages

# Load lightweight LLM
@st.cache_resource
def load_llm():

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    # model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto", device_map="auto")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
    
    return HuggingFacePipeline(pipeline=pipe)
    # pipe = pipeline("text-generation", model="google/flan-t5-small", max_new_tokens=256)
    # return HuggingFacePipeline(pipeline=pipe)
    
# extract 
def get_chunks(file_content):
    """
    split document into chunks
    """
    # initialise the recursive method
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=10
    )
    chunks = []

    for page in file_content:
        docs = [Document(page_content=page.page_content)]
        texts = splitter.split_documents(docs)
        chunks.append(texts)

    return list(itertools.chain(*chunks))


# Build retriever from uploaded content
@st.cache_resource
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
with st.expander("Try example questions"):
    for q in EXAMPLE_QUESTIONS:
        st.markdown(f"- {q}")

uploaded_file = st.file_uploader("Upload your file", type=["txt","pdf"])
if uploaded_file is not None:
    st.write("Filename:", uploaded_file.name)
    data = uploaded_file.read()

    if uploaded_file.type == "text/plain":
        # st.text_area("Content", data.decode("utf-8"), height=300)
        st.info("Uploaded txt file")
    else:
        st.info(f"Uploaded {len(data)} bytes (PDF or other format)")
query = st.text_input("Ask a question ")

if uploaded_file is not None:
    # st.success("file uploaded")
    if uploaded_file.type == "text/plain":
        # st.text_area("Content", data.decode("utf-8"), height=300)
        st.info("Uploaded txt file")
        docs = read_uploaded_file(uploaded_file)
    else:
        st.info(f"Uploaded {data} bytes (PDF or other format)")
        docs = st.pdf(uploaded_file.read())
    
    retriever = build_retriever(data.decode("utf-8"))
    llm = load_llm()

    # qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)

    if query:
        with st.spinner("Generating answer..."):
            result = qa_chain.invoke({"input":query})
            # st.info(result)
            answer =  result["answer"].split("\nAnswer:")[-1].strip()
        st.success(answer)

else:
    st.info("Please upload a `.txt or .pdf` file or use the sample provided.")
