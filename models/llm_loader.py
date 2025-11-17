"""LLM loading and initialization."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
import streamlit as st


@st.cache_resource
def load_qwen_llm(model_name, max_new_tokens=256, temperature=0.7, top_p=0.95):
    """
    Load Qwen LLM model.
    
    Args:
        model_name: HuggingFace model identifier
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        
    Returns:
        HuggingFacePipeline: Wrapped LLM for LangChain
    """
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        return_full_text=False
    )
    
    return HuggingFacePipeline(pipeline=pipe)
