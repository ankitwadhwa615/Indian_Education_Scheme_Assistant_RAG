# Indian Education Scheme Assistant

A portfolio-ready Retrieval-Augmented Generation (RAG) application for exploring Indian government education and training schemes in plain English. It retrieves relevant records from a local, source-derived knowledge base before asking an LLM to compose an answer, and exposes the retrieved records in the interface for review.

> This is an informational prototype. Scheme rules, dates, and eligibility can change; users should confirm important decisions on the linked official government portals.

## Highlights

- Answers questions on scheme benefits, eligibility, beneficiaries, and application pathways.
- Indexes **693** scheme-detail records from the included JSON dataset.
- Uses `BAAI/bge-small-en-v1.5` embeddings with local Chroma persistence.
- Uses Groq-hosted generation, with the model configurable through `GROQ_MODEL`.
- Supports multiple in-browser chat threads and preserves recent turn context for follow-up questions.
- Shows retrieved scheme records and chunks for transparency.
- Hides model reasoning so users receive only the final response.

## Architecture

```text
Government scheme JSON → normalize + chunk → BGE embeddings → Chroma vector store
                                                           ↓ top 5 chunks
Streamlit chat UI → LangChain grounded prompt → Groq LLM → Answer + sources
```

## Tech stack

| Area | Tools |
| --- | --- |
| UI | Streamlit |
| RAG orchestration | LangChain |
| Vector store | ChromaDB |
| Embeddings | Hugging Face `BAAI/bge-small-en-v1.5` |
| LLM inference | Groq via `langchain-groq` |
| Dataset acquisition | myScheme API export script |

## Getting started

Prerequisites: Python 3.10+ and a Groq API key.

```bash
git clone <your-repository-url>
cd RAG_Assistant
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=qwen/qwen3.6-27b  # optional
```

The generated `scheme_db` is intentionally excluded from Git. On a fresh deployment,
the app automatically creates it in temporary runtime storage from the included JSON
dataset. This can make the first answer slower. To build it locally in advance:

```bash
python ingest.py --reset
streamlit run app.py
```

The first embedding-model use may download model weights from Hugging Face.

### Deployment note

Keep `Education_scheme_details.json` committed to the repository. It is required for
first-run indexing in hosted environments; `scheme_db/` does not need to be committed.
The generated index is stored outside Streamlit's watched source folder so indexing
does not interrupt the running app.

## Project structure

```text
├── app.py                       # Streamlit chat experience
├── rag.py                       # Lazy-loaded retrieval and grounded LLM chain
├── ingest.py                    # JSON normalization and Chroma indexing CLI
├── fetch_Scheme_Data.py          # Optional myScheme API data-refresh script
├── Education_scheme_details.json # Included scheme detail dataset
├── scheme_db/                    # Local Chroma persistence (generated)
└── requirements.txt
```

## Design choices

- **Local ChromaDB:** simple local setup with no managed vector infrastructure.
- **Metadata-rich documents:** preserve scheme name, category, beneficiaries, ministry, and agency for better retrieval context.
- **Lazy initialization:** initializes model clients only after the first query, improving startup and error handling.
- **Grounding-first prompt:** instructs the model not to invent scheme facts beyond retrieved context.
- **Bounded chat context:** includes the latest eight messages for follow-ups without growing prompts indefinitely.

## Refreshing the dataset

Set `MY_SCHEME_API_KEY` in `.env`, then run:

```bash
python fetch_Scheme_Data.py
python ingest.py --reset
```

Review the refreshed JSON before deployment; upstream coverage and schema may change.

## Limitations and next steps

- The data is a point-in-time export, not a live policy feed.
- A reranker and similarity threshold would improve ambiguous queries.
- English is the primary supported language.
- A labelled evaluation set for retrieval recall and groundedness would be a valuable next improvement.

## Data acknowledgement

The dataset is derived from publicly available Indian government scheme information via myScheme for educational and portfolio demonstration purposes.
