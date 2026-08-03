"""Build the local Chroma vector database from the scheme JSON export."""

import argparse
import json
from pathlib import Path
import shutil

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Education_scheme_details.json"
DATABASE_DIR = BASE_DIR / "scheme_db"
READY_FILE = DATABASE_DIR / ".index_ready"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def build_documents(schemes: list[dict]) -> list[Document]:
    """Normalize nested API records into searchable, metadata-rich documents."""
    documents = []
    for scheme in schemes:
        en = (scheme.get("data") or {}).get("en") or {}
        basic, content = en.get("basicDetails") or {}, en.get("schemeContent") or {}

        def labels(items):
            return ", ".join(item.get("label", "") for item in (items or []) if item.get("label"))

        name = basic.get("schemeName", "Unnamed scheme")
        agency, category = basic.get("implementingAgency", ""), labels(basic.get("schemeCategory"))
        beneficiaries = labels(basic.get("targetBeneficiaries"))
        ministry = (basic.get("nodalMinistryName") or {}).get("label", "")
        department = (basic.get("nodalDepartmentName") or {}).get("label", "")
        applications = "\n".join(f"Mode: {item.get('mode', '')}\nURL: {item.get('url', '')}" for item in en.get("applicationProcess", []))
        references = "\n".join(f"{item.get('title', '')}: {item.get('url', '')}" for item in content.get("references", []))
        text = f"""Scheme Name: {name}
Level: {(basic.get('level') or {}).get('label', '')}
Tags: {', '.join(basic.get('tags', []))}
Categories: {category}
Subcategories: {labels(basic.get('schemeSubCategory'))}
Target Beneficiaries: {beneficiaries}
Implementing Agency: {agency}
Ministry: {ministry}
Department: {department}
Description: {content.get('detailedDescription_md') or content.get('briefDescription') or ''}
Benefits: {content.get('benefits_md') or ''}
Exclusions: {content.get('exclusions_md') or ''}
Application Process:\n{applications}
References:\n{references}"""
        documents.append(Document(page_content=text, metadata={
            "scheme_name": name, "agency": agency, "category": category,
            "beneficiaries": beneficiaries, "ministry": ministry,
        }))
    return documents


def is_database_ready() -> bool:
    """Check for indexed chunks, including databases built before the marker existed."""
    if READY_FILE.exists():
        return True
    if not DATABASE_DIR.exists():
        return False
    try:
        client = chromadb.PersistentClient(path=str(DATABASE_DIR))
        is_populated = any(collection.count() > 0 for collection in client.list_collections())
    except Exception:
        return False
    if is_populated:
        READY_FILE.write_text("ready\n", encoding="utf-8")
    return is_populated


def create_database(reset: bool = False):
    """Create the database from the bundled dataset and return its path."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}. Ensure Education_scheme_details.json is included in the deployment."
        )
    if DATABASE_DIR.exists():
        if not reset and is_database_ready():
            return DATABASE_DIR
        shutil.rmtree(DATABASE_DIR)
    with DATA_FILE.open(encoding="utf-8") as file:
        schemes = json.load(file)
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(build_documents(schemes))
    print(f"Loaded {len(schemes)} schemes and created {len(chunks)} chunks.")
    Chroma.from_documents(documents=chunks, embedding=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL), persist_directory=str(DATABASE_DIR))
    # Write this only after Chroma has accepted every chunk. A folder alone is
    # not proof of a usable index on ephemeral hosted environments.
    READY_FILE.write_text("ready\n", encoding="utf-8")
    print(f"Vector database created at {DATABASE_DIR}")
    return DATABASE_DIR


def main():
    parser = argparse.ArgumentParser(description="Build the local scheme vector database.")
    parser.add_argument("--reset", action="store_true", help="Replace an existing scheme_db before indexing; prevents duplicate chunks.")
    args = parser.parse_args()
    if DATABASE_DIR.exists() and is_database_ready() and not args.reset:
        raise FileExistsError(f"{DATABASE_DIR} already exists. Run `python ingest.py --reset` to rebuild it.")
    create_database(reset=args.reset)


if __name__ == "__main__":
    main()
