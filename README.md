
# Retrieval-Augmented Generation (RAG) with Pinecone and Gemini

## 📌 Project Overview

This project implements a modern Retrieval-Augmented Generation (RAG) system using:

- **HuggingFace Embeddings** (all-MiniLM-L6-v2)
- **Pinecone** as vector database
- **Google Gemini** as the Large Language Model (LLM)
- **LangChain (LCEL)** for orchestration

The system enhances LLM responses by retrieving relevant contextual information from a vector database before generating an answer.

---

## 🧠 What is RAG?

Retrieval-Augmented Generation (RAG) is an architecture that improves large language models by:

1. Retrieving relevant documents from an external knowledge base.
2. Injecting that context into the LLM.
3. Generating grounded, more accurate responses.

Unlike traditional LLMs, RAG reduces hallucinations and allows dynamic knowledge updates.

---

## 🏗 Architecture

### Pipeline

1. Document Loading
2. Text Chunking
3. Embedding Generation (384-dimensional vectors)
4. Storage in Pinecone (Cosine Similarity)
5. Retrieval of Top-K Relevant Chunks
6. Context-aware Answer Generation with Gemini

### Diagram

User → Retriever (Pinecone) → Context → Gemini LLM → Response

---

## 🛠 Technologies Used

- Python 3.11
- LangChain (modern LCEL)
- HuggingFace Sentence Transformers
- Pinecone (Serverless)
- Google Gemini (gemini-1.5-flash)

---

## 📦 Installation

### 1️⃣ Clone the repository

```bash
git clone <your_repo_link>
cd rag-project
````

### 2️⃣ Create virtual environment (recommended)

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

### Step 2 – Ask questions

```bash
python query.py
```

Example:

```
What is RAG?
```

---

## 📊 Why Pinecone?

Pinecone allows efficient similarity search across high-dimensional embeddings, enabling scalable retrieval in real-world AI systems.

---

## 📚 Why HuggingFace Embeddings?

We use `all-MiniLM-L6-v2`:

* 384-dimensional vectors
* Lightweight
* Efficient
* No API cost

---

## 🤖 Why Gemini?

Gemini generates context-aware responses using the retrieved document chunks, improving factual accuracy compared to standalone LLMs.

---

## 🎯 Key Learning Outcomes

* Understanding of vector embeddings
* Semantic similarity search
* Vector databases (Pinecone)
* Modern LangChain LCEL pipelines
* Building production-style RAG systems

---

## 📌 Conclusion

This project demonstrates a complete end-to-end RAG pipeline integrating retrieval and generation to enhance LLM responses with external knowledge.

```
