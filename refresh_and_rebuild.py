import os
import uuid
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from db import load_all_documents  

# Load env vars
load_dotenv()
COHERE_TOKEN = os.getenv("COHERE_API_KEY")

if not COHERE_TOKEN:
    raise ValueError("❌ Missing Cohere API Key. Check your .env file.")


print("🌐 Scraping Silvestre website for product and general info...")
docs = load_all_documents()
print("✅ Scraping complete")


print("⏳ Rebuilding vectorstore with Cohere embeddings...")

embeddings = CohereEmbeddings(
    cohere_api_key=COHERE_TOKEN,
    model="embed-multilingual-v3.0"
)
print("[INFO] Using Cohere for embedding")

num_products = sum(1 for d in docs if d.metadata.get("type") == "product")
num_general = len(docs) - num_products
print(f"[INFO] Loaded {num_products} product documents and {num_general} general documents for embedding.")

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    ids=[str(uuid.uuid4()) for _ in docs],
    persist_directory="chroma_db"
)

vectorstore.persist()
print("✅ Vectorstore rebuilt successfully and saved to chroma_db/")
