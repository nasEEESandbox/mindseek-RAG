import os
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("❌ OPENAI_API_KEY not set! Use export or .env file")

vector_db = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"))

def retrieve_docs(query, top_k=3):
    """Retrieve most relevant documents from ChromaDB."""
    results = vector_db.similarity_search(query, k=top_k)
    return [doc.page_content for doc in results]

if __name__ == "__main__":
    query = input("🔍 Enter your query: ")
    results = retrieve_docs(query)

    print("\n💡 Most Relevant Results:")
    for i, res in enumerate(results):
        print(f"\n[{i+1}] {res}\n{'-'*50}")
