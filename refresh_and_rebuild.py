import os
import uuid
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from db import load_products_from_db, refresh_database  

# Load env vars
load_dotenv()
COHERE_TOKEN = os.getenv("COHERE_API_KEY")

if not COHERE_TOKEN:
    raise ValueError("Missing Cohere API Key")

# STEP 1: Refresh scraped data
print("🌐 Scraping Silvestre website and refreshing database...")
refresh_database()
print("✅ Database updated (silvestre_products.db)")

# STEP 2: Rebuild vectorstore
print("⏳ Rebuilding vectorstore with Cohere embeddings...")

embeddings = CohereEmbeddings(
    cohere_api_key=COHERE_TOKEN,
    model="embed-multilingual-v3.0"
)
print("[INFO] Using Cohere for embedding")

docs = load_products_from_db()

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    ids=[str(uuid.uuid4()) for _ in docs],
    persist_directory="chroma_db"
)

print("✅ Vectorstore rebuilt successfully and saved to chroma_db/")
