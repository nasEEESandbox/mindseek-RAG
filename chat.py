"""
This is a simple chatbot that uses a retrieval-based QA system to answer questions.
Not used as of right now, might be used in the future.
"""

import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

# Load API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not set! Use export or .env file")

# Load the vector database
persist_directory = "./chroma_db"
vector_db = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"))

# Create a retriever to search relevant chunks
retriever = vector_db.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 chunks

# Use GPT-4o to generate responses
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

# Create a RetrievalQA chain (Search + GPT Response)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

def ask_question(query):
    """Ask a question to the chatbot and return the answer"""
    response = qa_chain.invoke({"query": query})
    return response

# Interactive Chat Loop
if __name__ == "__main__":
    print("🤖 Chatbot is ready! Type 'exit' to stop.")
    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            print("👋 Goodbye!")
            break
        response = ask_question(query)
        print(f"🤖 Bot: {response}")
