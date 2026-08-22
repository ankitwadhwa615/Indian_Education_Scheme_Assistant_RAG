"""Retrieval and answer generation for the Scheme Assistant."""

from functools import lru_cache
from pathlib import Path
import os
import re

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "scheme_db"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

GREETINGS = {"hi", "hello", "hey", "good morning", "good evening", "good afternoon"}
ABOUT_ASSISTANT = {
    "who are you",
    "what are you",
    "what can you do",
    "how can you help",
    "help",
}


def expand_query(question: str) -> str:
    """Map common conversational abbreviations to terms present in the dataset."""
    expanded = re.sub(r"\bpost[- ]?grad\b", "post graduate", question, flags=re.IGNORECASE)
    if expanded != question:
        return f"{expanded} scholarships higher education post matric"
    return question

prompt = ChatPromptTemplate.from_template(
    """
You are the Indian Government Education and Training Scheme Assistant.

Answer only questions about Indian government education and training schemes.
Use the supplied context as the factual source of truth. Do not invent scheme
details, eligibility, benefits, or application steps. If the answer is not in
the context, say exactly: "I could not find relevant information in the scheme database."
Give a concise, clear answer; use bullets when they improve readability.

For broad discovery questions (for example, asking for postgraduate schemes),
identify the relevant scheme names found in the context and briefly explain why
each is relevant. Do not return the no-result message when the context contains
related schemes. Keep details attached to the correct scheme; retrieved chunks
can come from different schemes.

Conversation history:
{history}

Context:
{context}

Question: {question}
Answer:
"""
)


@lru_cache(maxsize=1)
def get_retriever():
    if not DATABASE_DIR.exists():
        raise RuntimeError(
            f"Vector database not found at {DATABASE_DIR}. "
            "Run ingest.py before deployment."
        )
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = Chroma(
        persist_directory=str(DATABASE_DIR),
        embedding_function=embeddings
    )

    count = vectorstore._collection.count()
    print(f"Loaded Chroma DB with {count} chunks")

    if count == 0:
        raise RuntimeError(
            f"Chroma database at {DATABASE_DIR} is empty."
        )

    return vectorstore.as_retriever(search_kwargs={"k": 5})


@lru_cache(maxsize=1)
def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured. Add it to your environment or .env file.")
    reasoning_options = {}
    if GROQ_MODEL.startswith("qwen/"):
        # Prevent Qwen reasoning tokens from appearing as <think> text in the UI.
        reasoning_options = {"reasoning_format": "hidden", "reasoning_effort": "none"}
    return ChatGroq(model=GROQ_MODEL, temperature=0.2, groq_api_key=api_key, **reasoning_options)


def get_answer(question: str, history: str = "") -> dict:
    """Return a grounded response and the documents used to produce it."""
    question = question.strip()
    if not question:
        return {"answer": "Please enter a question about a scheme.", "sources": []}
    question_clean = question.lower().strip().rstrip("?.!")
    if question_clean in GREETINGS:
        return {
            "answer": "Hello! I can help you explore Indian government education and training schemes, including benefits, eligibility, and application steps.",
            "sources": [],
        }
    if question_clean in ABOUT_ASSISTANT:
        return {
            "answer": (
                "I’m the Indian Government Education and Training Scheme Assistant. "
                "I can help you find information about scholarships, skill-development "
                "programmes, eligibility, benefits, and application processes in the local scheme database."
            ),
            "sources": [],
        }

    docs = get_retriever().invoke(expand_query(question))
    context = "\n\n".join(doc.page_content for doc in docs)
    if len(context.strip()) < 50:
        return {"answer": "I could not find relevant information in the scheme database.", "sources": docs}

    response = (prompt | get_llm()).invoke(
        {"question": question, "context": context, "history": history or "No prior conversation."}
    )
    answer = response.content
    if "i could not find relevant information in the scheme database" in answer.lower():
        # A broad query can retrieve useful records even when the model declines
        # to compose a detailed answer. Surface those verified scheme names.
        scheme_names = list(dict.fromkeys(
            document.metadata.get("scheme_name", "") for document in docs
            if document.metadata.get("scheme_name")
        ))
        if scheme_names:
            answer = (
                "I found these potentially relevant schemes in the database:\n\n"
                + "\n".join(f"- {name}" for name in scheme_names[:5])
                + "\n\nAsk about one of these schemes for eligibility, benefits, or application details."
            )
    return {"answer": answer, "sources": docs}
