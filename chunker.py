from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text_with_page(text, chunk_size=1000, chunk_overlap=100, page_delimiter="%%PAGE%%"):
    """
    Splits text that contains page delimiters into (chunk, page_number) tuples.
    """
    pages = text.split(page_delimiter)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks_with_page = []
    for i, page in enumerate(pages, start=1):
        chunks = text_splitter.split_text(page)
        for chunk in chunks:
            chunks_with_page.append((chunk, i))
    return chunks_with_page

if __name__ == "__main__":
    with open("extracted_text.txt", "r", encoding="utf-8") as file:
        text = file.read()
    
    chunks_with_page = split_text_with_page(text)
    print(f"✅ {len(chunks_with_page)} chunks created with page numbers")
