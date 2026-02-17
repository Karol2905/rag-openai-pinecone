import os
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

# Load document
loader = TextLoader("data/sample_document.txt")
documents = loader.load()

# Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
docs = splitter.split_documents(documents)

# HuggingFace embeddings (384 dim)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Connect vector store
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

# Upload documents
vectorstore.add_documents(docs)

print("✅ Documents indexed successfully.")
