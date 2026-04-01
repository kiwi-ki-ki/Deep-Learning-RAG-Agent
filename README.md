# Deep Learning RAG Agent

This project is a Retrieval-Augmented Generation (RAG)–powered interview preparation agent designed to help users study and understand core deep learning concepts through an interactive system.

The agent ingests structured study material, stores it in a vector database, and retrieves relevant information to generate grounded, explainable responses with source citations.

### Overview
This system was built to demonstrate a complete end-to-end RAG pipeline, including:

- Document ingestion and preprocessing
- Structured chunking of technical content
- Vector embedding and storage using ChromaDB
- Semantic retrieval based on user queries
- Controlled response generation using a language model
- A user interface for interacting with the system

The goal is to provide accurate, transparent answers grounded strictly in curated study material rather than relying on general model knowledge.

### Features
- Document Ingestion
  - Upload individual files or ingest an entire corpus
  - Automatically processes and stores documents in the vector database
- Duplicate Detection
  - Prevents re-ingestion of previously uploaded documents
- Chunk-Based Retrieval
  - Retrieves specific, relevant pieces of information instead of full documents
- Source Citations
  - All responses include references to the original content
- Hallucination Guard
  - Prevents the model from answering questions outside the provided corpus
- Interactive UI
  - Built with Streamlit
  - Includes document ingestion, document viewer, and chat interface

### Tech Stack
LangChain — retrieval and prompt orchestration
LangGraph — agent state management and flow control
ChromaDB — vector database for embeddings and retrieval
Streamlit — user interface
Python — core implementation

### Project Structure
deep-learning-rag-agent/
├── data/
│   └── corpus/            # Structured study material (ANN, CNN, RNN)
├── src/rag_agent/
│   ├── vectorstore/       # Vector database logic (ingestion, retrieval)
│   ├── agent/             # Prompts, state management, LangGraph flow
│   └── ui/                # Streamlit application
├── tests/                 # Unit tests
└── README.md

### How It Works
1. Study materials are manually created and structured into chunks
2. Each chunk is embedded into a vector representation
3. Embeddings are stored in ChromaDB
4. A user query is converted into a vector
5. The system retrieves the most relevant chunks
6. LangChain combines the retrieved content with prompts
7. The language model generates a grounded response with citations
8. Demo Capabilities

### The system supports:
- Ingesting documents through the UI
- Detecting and preventing duplicate uploads
- Answering technical deep learning questions
- Providing source-backed responses
- Rejecting off-topic queries

### Author
This project was completed individually by Kiera Wingo.

Generative AI tools, including ChatGPT and Claude, were used to assist with debugging and error resolution. All final implementation decisions and integrations were completed independently.

### Notes
This project was developed as part of a system design and machine learning workflow exercise, with a focus on building a functional and explainable RAG-based application.
