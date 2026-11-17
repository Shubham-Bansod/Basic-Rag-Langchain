# PDF RAG with LangChain and Gemini

A small command-line retrieval-augmented generation (RAG) application for asking questions about PDF files. It ingests PDFs into a local Chroma database, retrieves relevant passages with BM25 and vector search, compresses the retrieved context, and asks Gemini to answer from that context.

## Features

- Loads PDF files from `documents/` and splits them into overlapping text chunks.
- Stores chunks and metadata in a persistent local Chroma collection.
- Combines BM25 keyword ranking with Gemini embedding-based vector similarity.
- Filters retrieved chunks with `EmbeddingsFilter`, then uses an LLM extractor to keep question-relevant text.
- Answers questions with Gemini 2.5 Flash and a Pydantic structured response.
- Validates returned source IDs against retrieved chunks, then displays source PDF names and page numbers when available.
- Separates system instructions from the user question and retrieved context, and treats retrieved content as untrusted.

## Project layout

```text
.
├── documents/       # Source PDF files
├── chroma_db/       # Local vector database created by ingestion.py
├── ingestion.py     # PDF loading, chunking, and Chroma storage
├── query.py         # Hybrid retrieval, contextual compression, and Q&A
├── requirements.txt
└── .env             # Local API key configuration (create this)
```

## Requirements

- Python 3.10 or later
- A Google Gemini API key

Create and activate a virtual environment, then install the project requirements and the integrations used by the scripts:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install langchain-community langchain-chroma langchain-text-splitters langchain-classic rank_bm25 pypdf
```

On Windows PowerShell, activate the environment with:

```powershell
venv\Scripts\Activate.ps1
```

Create a `.env` file in the project root and add your Gemini API key:

```dotenv
GOOGLE_API_KEY=your_google_api_key
```

Keep `.env` private. Do not commit API keys or other secrets to Git.

## Use

1. Put one or more PDF files in `documents/`.
2. Ingest the PDFs and build the local Chroma collection:

   ```bash
   python ingestion.py
   ```

3. Start the question-answering CLI:

   ```bash
   python query.py
   ```

4. Enter a question. Type `exit` or `quit` to stop.

`ingestion.py` replaces the existing contents of the `pdf_documents` collection each time it runs. Keep your source PDFs; the Chroma database is generated data and can be rebuilt by running ingestion again.

## Retrieval flow

```text
PDFs → chunking → Chroma
                  ├─ BM25 keyword search ─┐
                  └─ vector similarity ───┴→ rank fusion
                                             → EmbeddingsFilter
                                             → LLM context extraction
                                             → structured Gemini response
                                             → validate source IDs
```

The two retrievers use equal weights. They each fetch up to eight candidates. `EmbeddingsFilter` retains up to three relevant chunks, then `LLMChainExtractor` extracts relevant passages from those chunks. Compression can add model and embedding calls, increasing latency and cost; the extractor may also omit qualifications, so verify important answers against the cited source PDFs.

The answer model returns a `RAGResponse` object with these fields:

| Field | Type | Purpose |
| --- | --- | --- |
| `answer` | `str` | Answer grounded in the compressed excerpts |
| `found` | `bool` | Whether the excerpts contain an answer |
| `source_ids` | `list[int]` | One-based IDs of excerpts supporting the answer |

The application maps valid source IDs back to the retrieved documents' metadata. IDs outside the retrieved set are discarded, and source references are suppressed when `found` is false. The model does not supply source filenames or page numbers; those come from the documents' metadata.

## Reducing prompt injection risk

Retrieved PDF text is untrusted input. A document can contain malicious or misleading instructions, so retrieval and compression must never grant it authority over the application.

1. Clearly separate system instructions, user questions, and retrieved context. `query.py` uses separate system and human messages and labels excerpts as untrusted data.
2. Tell the model to treat retrieved content as untrusted data. The system message instructs it to answer only from the excerpts and ignore instructions contained in them.
3. Restrict retrieval to documents the authenticated user is authorized to access. This example is a local, single-user CLI and does not implement user authentication or document-level access control. Add authorization filtering before using it for multiple users or sensitive documents.
4. Never expose API keys, secrets, database credentials, or internal system prompts to the model unnecessarily. Keep credentials in environment configuration and out of retrieved documents and prompts.
5. Validate model outputs before performing sensitive actions. The response is parsed into the `RAGResponse` schema and source IDs are checked against retrieved chunks. This application only prints answers; if you add actions, validate their inputs and permissions independently of model output.
6. Do not allow retrieved text to authorize tool calls or override application permissions. Enforce access controls and tool permissions in application code.

## Troubleshooting

- **`No documents found. Run ingestion.py first.`** Add PDFs to `documents/` and run `python ingestion.py`.
- **Missing module errors:** confirm the virtual environment is active, then install the packages listed above.
- **Gemini authentication errors:** confirm `GOOGLE_API_KEY` is set in `.env` and that the key is valid.
- **Unexpected answers:** inspect the printed source names and compare the response with the source PDF. Retrieval and LLM compression can miss relevant passages.

## Notes

- The local database path is `./chroma_db`, relative to the current working directory.
- PDF category metadata is inferred from filenames in `ingestion.py` (`HR`, `SECURITY`, `FINANCE`, or `GENERAL`).
- This is a learning project. Add authentication, per-user access controls, monitoring, and stronger output validation before exposing it as a shared service.
