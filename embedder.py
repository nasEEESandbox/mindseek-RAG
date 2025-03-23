import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from chunker import split_text_with_page  # use the modified splitter that returns (chunk, page)

# Load environment variables from .env
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not set! Use export or .env file")
print("✅ API Key loaded successfully!")

# Read extracted text from the file (assumed to include page delimiters)
with open("extracted_text.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Split the extracted text into chunks using the new split_text_with_page function
chunks_with_page = split_text_with_page(text)
print(f"✅ {len(chunks_with_page)} chunks created with page numbers")

# Create Document objects from the chunks with page metadata
documents = []
for chunk_text, page in chunks_with_page:
    documents.append(Document(
        page_content=chunk_text,
        metadata={"source": "DSM-5", "page": page}
    ))

# Initialize OpenAI Embeddings (Using text-embedding-3-large)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Store the documents in VectorDB (ChromaDB), preserving metadata
vector_db = Chroma.from_documents(documents, embedding=embeddings, persist_directory="./chroma_db")

print("✅ Embeddings stored successfully!")
