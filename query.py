
from dotenv import load_dotenv

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


# Load environment variables
load_dotenv()

CHROMA_DB_DIR = "./chroma_db"

# 1. Initialize Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# 2. Connect to existing ChromaDB
vector_store = Chroma(
    collection_name="pdf_documents",
    embedding_function=embeddings,
    persist_directory=CHROMA_DB_DIR
)


# 3. Create retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# 4. Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# 5. Create prompt template
prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful AI assistant.

    Answer the question using ONLY the
    information provided in the context.

    If the answer is not present in the context,
    say that you could not find the answer
    in the provided documents.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)


# 6. Create RAG chain
rag_chain = prompt | llm


# 7. Ask questions
def ask_question(question):

    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        print("No relevant documents found.")
        return

    # Combine retrieved document content
    context = "\n\n".join(
        document.page_content
        for document in retrieved_docs
    )

    # Pass context and question to Gemini
    response = rag_chain.invoke({
        "context": context,
        "question": question
    })

    print("\nAnswer:")
    print(response.content)

    # Display sources
    print("\nSources:")

    for document in retrieved_docs:

        source = document.metadata.get("source", "Unknown")
        page = document.metadata.get("page", "Unknown")

        print(f"PDF: {source} | Page: {page + 1 if isinstance(page, int) else page}")


# 8. Terminal application
def main():

    print("\n===== LANGCHAIN RAG APPLICATION =====")
    print("Ask questions about your PDF documents.")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("You: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            ask_question(question)

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
