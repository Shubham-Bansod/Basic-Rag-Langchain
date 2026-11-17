
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
# retriever = vector_store.as_retriever(
#     search_type="mmr",
#     search_kwargs={
#         "k": 3,
#         "fetch_k": 10,
#         "lambda_mult": 0.5
#     }
# )

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.5,
        "k": 3
    }
)

# retriever = vector_store.as_retriever(
#     search_type="similarity",
#     search_kwargs={
#         "k": 3,
#         "filter": {
#             "$and": [
#                 {"category": "HR"},
#                 {"source": "leave_policy.pdf"}
#             ]
#         }
#     }
# )
# retriever = vector_store.as_retriever(
#     search_type="similarity",
#     search_kwargs={
#         "k": 5,
#         "filter": {
#             "$and": [
#                 {"source": "leave_policy.pdf"},
#                 {"page": {"$gte": 9}},
#                 {"page": {"$lte": 19}}
#             ]
#         }
#     }
# )

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

# Dynamic retrieval with metadata filtering

def ask_question(question, category=None, source=None):

    # Build metadata filter
    filters = []

    if category:
        filters.append({
            "category": category
        })

    if source:
        filters.append({
            "source": source
        })

    # Combine filters
    metadata_filter = None

    if len(filters) == 1:
        metadata_filter = filters[0]

    elif len(filters) > 1:
        metadata_filter = {
            "$and": filters
        }

    # Retrieve relevant documents
    retrieved_docs = vector_store.similarity_search(
        query=question,
        k=3,
        filter=metadata_filter
    )

    if not retrieved_docs:
        print("No matching documents found.")
        return

    # Prepare context
    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    # Generate answer
    response = rag_chain.invoke({
        "context": context,
        "question": question
    })

    print("\nAnswer:")
    print(response.content)

    print("\nSources:")

    for doc in retrieved_docs:
        print({
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
            "category": doc.metadata.get("category")
        })

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
