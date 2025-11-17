"""Utilities for processing uploaded documents."""

import io

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader


def read_uploaded_file(uploaded_file):
    """
    Read and process uploaded file (TXT or PDF).
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        list: List of text chunks from the document
    """

    uploaded_file.seek(0)
    
    if uploaded_file.type == "application/pdf":
        return process_pdf(uploaded_file)
    else:
        return process_text(uploaded_file)


def process_pdf(uploaded_file):
    """Extract text from PDF file."""
    pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return split_into_chunks(text)


def process_text(uploaded_file):
    """Read text file."""
    text = uploaded_file.read().decode("utf-8")
    return split_into_chunks(text)


def split_into_chunks(text):
    """Split text into chunks by lines."""
    docs = text.split("\n")
    docs = [doc.strip() for doc in docs if doc.strip()]
    return docs
