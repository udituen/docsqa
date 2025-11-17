---
title: DocsQA
emoji: 📚
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: Upload a document and ask questions based on its content
---

# Welcome to DocsQA!


```
project/
├── app.py                      # Main entry point
├── config.py                   # Configuration settings
├── utils/
│   └── document_processor.py  # Document reading & processing
├── models/
│   ├── llm_loader.py          # Qwen LLM loading
│   └── retriever.py           # FAISS retriever setup
├── chains/
│   └── qa_chain.py            # QA chain creation
└── ui/
    ├── sidebar.py             # Sidebar components
    └── chat.py                # Chat interface
```

