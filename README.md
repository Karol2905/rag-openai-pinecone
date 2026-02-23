
# Retrieval-Augmented Generation (RAG) with Pinecone and Gemini

## 📌 Project Overview

This project implements a modern Retrieval-Augmented Generation (RAG) system using:

- HuggingFace Embeddings (all-MiniLM-L6-v2)
- Pinecone (vector database)
- Google Gemini (LLM)
- LangChain (LCEL architecture)

The system enhances LLM responses by retrieving relevant contextual information from a vector database before generating an answer.

---

## 🧠 What is RAG?

Retrieval-Augmented Generation (RAG) is an architecture that improves LLM performance by:

1. Retrieving relevant information from an external knowledge base.
2. Injecting retrieved context into the LLM.
3. Generating grounded and context-aware responses.

This reduces hallucinations and allows dynamic knowledge updates.

---

## 🏗 Architecture

### Pipeline

1. Document Loading
2. Text Chunking
3. Embedding Generation (384-dimensional vectors)
4. Storage in Pinecone (Cosine Similarity)
5. Similarity Retrieval
6. Context-Aware Generation with Gemini

### Diagram

User → Retriever (Pinecone) → Retrieved Context → Gemini LLM → Response

---

## 🛠 Technologies Used

- Python 3.11
- LangChain (modern LCEL)
- HuggingFace Sentence Transformers
- Pinecone (Serverless)
- Google Gemini (gemini-1.5-flash)

---

## 📂 Project Structure

````

rag-project/
│
├── ingest.py        # Index documents into Pinecone
├── query.py         # Query the RAG system
├── data/
│   └── sample_document.txt
├── requirements.txt
├── .env.example
└── README.md

````

---

## 📦 Installation

### 1️⃣ Clone the repository

```bash
git clone <repo_link>
cd rag-project
````

### 2️⃣ Create virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=rag-hf
GOOGLE_API_KEY=your_google_api_key
```

---

## 🚀 Running the Project

### Step 1 – Index documents

```bash
python ingest.py
```

This will:

* Load documents
* Split them into chunks
* Generate embeddings
* Store them in Pinecone

---

### Step 2 – Query the system

```bash
python query.py
```

Example question:

```
What is RAG?
```

---

## 📊 Why Pinecone?

Pinecone enables efficient similarity search across high-dimensional embeddings, making retrieval scalable and fast.

---

## 📚 Why HuggingFace Embeddings?

We use `all-MiniLM-L6-v2` because:

* It generates 384-dimensional vectors
* It is lightweight and efficient
* It requires no API cost

---

## 🤖 Why Gemini?

Gemini generates context-aware responses based on retrieved documents, improving accuracy compared to standalone LLMs.

---

## 🎯 Key Learning Outcomes

* Understanding semantic embeddings
* Vector similarity search
* Vector databases
* Retrieval-Augmented Generation
* Modern LangChain architecture (LCEL)

---

## 📌 Conclusion

This project demonstrates a complete end-to-end RAG pipeline integrating retrieval and generation to enhance LLM responses with external knowledge.
