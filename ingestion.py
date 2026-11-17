
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter,sentence_based_chunking
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

def get_category(filename):

    filename = filename.lower()

    if "employee" in filename or "leave" in filename:
        return "HR"

    elif "security" in filename:
        return "SECURITY"

    elif "finance" in filename:
        return "FINANCE"

    return "GENERAL"

# 1. Load PDF documents
def load_pdfs():

    documents = []

    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

    for pdf_path in pdf_files:

        loader = PyPDFLoader(str(pdf_path))

        pdf_documents = loader.load()

        category = get_category(pdf_path.name)

        for document in pdf_documents:

            document.metadata["source"] = pdf_path.name

            document.metadata["category"] = category

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

def compare_chunking(documents):

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    recursive_chunks = recursive_splitter.split_documents(
        documents
    )

    sentence_chunks = sentence_based_chunking(documents)

    print("Recursive chunks:", len(recursive_chunks))
    print("Sentence chunks:", len(sentence_chunks))

    print("\nRecursive example:")
    print(recursive_chunks[0].page_content)

    print("\nSentence example:")
    print(sentence_chunks[0].page_content)

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

    print("\nChecking stored metadata:")

    for item in vector_store.get(
        limit=5,
        include=["metadatas"]
    )["metadatas"]:

        print(item)

    # Add new document chunks
    vector_store.add_documents(chunks)

    print(f"Successfully stored {len(chunks)} chunks in ChromaDB!")


# 4. Main ingestion pipeline
def main():

    print("\nStarting PDF ingestion...\n")

    documents = load_pdfs()

    compare_chunking(documents)

    if not documents:
        return

    print(f"\nTotal pages loaded: {len(documents)}")

    chunks = split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    store_in_chroma(chunks)

    print("\nIngestion completed successfully!")


if __name__ == "__main__":
    main()
