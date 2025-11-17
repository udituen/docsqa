"""Configuration settings for the RAG application."""

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


# Model configurations
QWEN_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Generation parameters
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.95

