import os
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

# Embeddings (same as ingest)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

retriever = vectorstore.as_retriever()

# Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:

{context}

Question: {question}
""")

# RAG pipeline
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

query = input("Ask a question: ")

docs = retriever.invoke(query)
context = format_docs(docs)

chain = prompt | llm | StrOutputParser()

response = chain.invoke({
    "context": context,
    "question": query
})

print("\nAnswer:")
print(response)
