# vectorstore_utils.py

import os
import uuid
import hashlib
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document
from db import load_all_documents

# === Constants ===
CHROMA_PATH = "chroma_db"
load_dotenv()
COHERE_TOKEN = os.getenv("COHERE_API_KEY")

# === Utility Functions ===
def compute_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def load_vectorstore():
    embeddings = CohereEmbeddings(
        cohere_api_key=COHERE_TOKEN,
        model="embed-multilingual-v3.0"
    )
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

# === Build Function ===
def build_vectorstore_if_new():
    if not COHERE_TOKEN:
        raise ValueError("❌ Missing Cohere API Key. Check your .env file.")

    embeddings = CohereEmbeddings(
        cohere_api_key=COHERE_TOKEN,
        model="embed-multilingual-v3.0"
    )

    print("🌐 Scraping Silvestre website for product and general info...")
    new_docs = load_all_documents()
    print(f"📄 Loaded {len(new_docs)} documents")

    # Remove empty
    new_docs = [d for d in new_docs if d.page_content.strip()]
    if not new_docs:
        print("❌ No usable documents found after filtering. Aborting.")
        return None

    # Compute hashes for new docs
    new_hashes = set(compute_hash(d.page_content) for d in new_docs)

    # Load existing hashes from vectorstore
    if Path(f"{CHROMA_PATH}/index").exists():
        existing_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        raw = existing_store._collection.get(include=["documents"])
        existing_hashes = set(compute_hash(doc) for doc in raw["documents"])
    else:
        existing_store = None
        existing_hashes = set()

    # Find only new docs
    unique_docs = [doc for doc in new_docs if compute_hash(doc.page_content) not in existing_hashes]

    num_products = sum(1 for d in unique_docs if d.metadata.get("type") == "product")
    num_general = len(unique_docs) - num_products
    print(f"🔢 New docs to embed: {len(unique_docs)} ({num_products} products, {num_general} general)")

    if not unique_docs:
        print("📭 No new documents to add. Vectorstore unchanged.")
        return existing_store or load_vectorstore()

    # Add to or create vectorstore
    if existing_store:
        print("📥 Adding to existing vectorstore...")
        existing_store.add_documents(
            documents=unique_docs,
            ids=[str(uuid.uuid4()) for _ in unique_docs]
        )
        existing_store.persist()
        vectorstore = existing_store
    else:
        print("📦 Creating new Chroma vectorstore...")
        Chroma.from_documents(
            documents=unique_docs,
            embedding=embeddings,
            ids=[str(uuid.uuid4()) for _ in unique_docs],
            persist_directory=CHROMA_PATH
        )
        vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    print(f"✅ Vectorstore updated and saved to `{CHROMA_PATH}/`")

    raw = vectorstore._collection.get(include=["documents", "metadatas"])
    print("📦 Final saved chunk count:", len(raw["documents"]))
    types = Counter(m.get("type", "unknown") for m in raw["metadatas"])
    print("📊 Chunk types:", dict(types))
    
    # Optional: hot reload for rag_chain
    try:
        from rag_chain import reload_vectorstore
        reload_vectorstore()
    except Exception as e:
        print("[WARN] Could not reload vectorstore in rag_chain:", e)


    return vectorstore
