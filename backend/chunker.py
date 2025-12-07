from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Chunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 75):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, document: Document):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, length_function=len, separators=["\n\n", "\n", " ", ""])
        chunks = text_splitter.split_documents(document)
        return chunks