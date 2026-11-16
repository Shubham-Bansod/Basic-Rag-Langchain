
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


# Load environment variables
load_dotenv()

# Project paths
DOCUMENTS_DIR = Path("./documents")
CHROMA_DB_DIR = "./chroma_db"

# Gemini embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# 1. Load PDF documents
def load_pdfs():

    documents = []

    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found!")
        return documents

    for pdf_path in pdf_files:

        print(f"Loading: {pdf_path.name}")

        loader = PyPDFLoader(str(pdf_path))

        pdf_documents = loader.load()

        # Add source filename to metadata
        for document in pdf_documents:
            document.metadata["source"] = pdf_path.name

        documents.extend(pdf_documents)

    return documents


# 2. Split documents into chunks
def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


# 3. Store documents in ChromaDB
def store_in_chroma(chunks):

    if not chunks:
        print("No chunks to store!")
        return

    # Remove previous collection and recreate it
    vector_store = Chroma(
        collection_name="pdf_documents",
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

    # Delete old records from this collection
    existing_ids = vector_store.get()["ids"]

    if existing_ids:
        vector_store.delete(ids=existing_ids)

    # Add new document chunks
    vector_store.add_documents(chunks)

    print(f"Successfully stored {len(chunks)} chunks in ChromaDB!")


# 4. Main ingestion pipeline
def main():

    print("\nStarting PDF ingestion...\n")

    documents = load_pdfs()

    if not documents:
        return

    print(f"\nTotal pages loaded: {len(documents)}")

    chunks = split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    store_in_chroma(chunks)

    print("\nIngestion completed successfully!")


if __name__ == "__main__":
    main()
