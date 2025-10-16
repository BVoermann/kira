from langchain.document_loaders import pyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os

from langchain_core.vectorstores import VectorStore


class DocumentProcessor:
    def __init__(self, persist_directory="./chroma_db"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence_transformers/all-MiniLM-L6-v2"
        )
        self.persist_directory = persist_directory
        self.vectorstore = None

    def load_documents(self, file_paths):
        """Load documents"""
        documents = []

        for file_path in file_paths:
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith(".txt"):
                loader = TextLoader(file_path)
            else:
                print(f"Unsupported file type: {file_path}")
                continue

            documents.extend(loader.load())

        return documents

    def split_documents(self, documents):
        """Split documents"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        return chunks

    def create_vectorstore(self, chunks):
        """Create vector database from chunks"""
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embeddings=self.embeddings,
            persist_directory=self.persist_directory
        )
        self.vectorstore.persist()
        print(f"Created vectorstore with {len(chunks)} chunks")

    def load_vectorstore(self):
        """Load existing vectorstore"""
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

    def process_documents(self, file_paths):
        """Main pipeline: load -> split -> vectorize"""
        documents = self.load_documents(file_paths)
        chunks = self.split_documents(documents)
        self.create_vectorstore(chunks)