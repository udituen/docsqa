"""Document retrieval system."""

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def build_retriever(docs, embedding_model_name="all-MiniLM-L6-v2"):
    """
    Build FAISS retriever from documents.
    
    Args:
        docs: List of text documents
        embedding_model_name: Name of the embedding model
        
    Returns:
        Retriever object
    """
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    db = FAISS.from_texts(docs, embeddings)
    return db.as_retriever()
