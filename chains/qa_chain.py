"""Conversational QA chain setup."""

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


def create_qa_chain(llm, retriever):
    """
    Create conversational QA chain with memory.
    
    Args:
        llm: Language model
        retriever: Document retriever
        
    Returns:
        ConversationalRetrievalChain
    """
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
    
    return chain
