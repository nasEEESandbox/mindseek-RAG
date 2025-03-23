import os
import PyPDF2
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_documents_from_pdf(pdf_path, output_txt="extracted_text.txt", page_delimiter="\n%%PAGE%%\n"):
    """
    Extracts text from each page of a text-based PDF, writes the extracted text (with page delimiters)
    to a text file, splits each page's text using RecursiveCharacterTextSplitter, and returns a list of
    Document objects with metadata including the source filename and page number.
    """
    pages_text = []
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for i, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                pages_text.append((i, page_text))
    
    full_text = page_delimiter.join([f"Page {i}:\n{text}" for i, text in pages_text])
    with open(output_txt, "w", encoding="utf-8") as txt_file:
        txt_file.write(full_text)
    print(f"✅ Extracted text written to {output_txt}")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n%%PAGE%%\n", "\n\n", "\n", " "]
    )
    
    documents = []
    for page_num, page_text in pages_text:
        chunks = splitter.split_text(page_text)
        for chunk in chunks:
            documents.append(Document(
                page_content=chunk,
                metadata={"source": os.path.basename(pdf_path), "page": page_num}
            ))
    return documents

if __name__ == "__main__":
    pdf_path = "DSM-5.pdf"  # Change this to your PDF path
    docs = extract_documents_from_pdf(pdf_path)
    print(f"✅ Extracted {len(docs)} document chunks from {pdf_path}")
