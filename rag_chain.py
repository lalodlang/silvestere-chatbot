
import os
import re
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from difflib import SequenceMatcher
from live_scraper import crawl_product_pages, scrape_product_page
import chromadb
from db import GENERAL_PAGES 
from db import get_all_documents, chunk_documents
from intent_utils import is_followup_question

load_dotenv()

# Global state
last_product_doc = None

# Constants
CHROMA_PATH = "chroma_db"
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-70b-8192"

# Embeddings
def get_embeddings_model():
    return CohereEmbeddings(model="embed-multilingual-v3.0", cohere_api_key=COHERE_API_KEY)

embedding = get_embeddings_model()
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k":5})  # k = number of context chunks


# LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model=GROQ_MODEL,
    temperature=0 # I set this to 0 to avoid hallucinations
)

# Prompts
PRODUCT_PROMPT = PromptTemplate.from_template("""
You are a corporate assistant for Silvestre Oil Company.

Only use the following context to answer. If it's missing or off-topic, say:
"I'm sorry, I can only answer questions about our official products."

<context>
{context}
</context>

<chat_history>
{history}
</chat_history>

User: {question}

Respond concisely, with product name, price, availability, and raw URL.
""")

GENERAL_PROMPT = PromptTemplate.from_template("""
You are a professional assistant for Silvestre Oil Company.

Only use this official context to answer. Reply in a corporate tone. Provide link to information provided if available.
                                              
If not found in context, say:
"I'm sorry, I can only provide information from our website."

<context>
{context}
</context>

<chat_history>
{history}
</chat_history>

User: {question}
""")

# Chains
rag_chain_product = (
    RunnableParallel(
        context=RunnableLambda(lambda x: x["context"]),
        question=RunnablePassthrough(),
        history=RunnableLambda(lambda x: x.get("history", ""))
    )
    | PRODUCT_PROMPT
    | llm
    | StrOutputParser()
)

rag_chain_general = (
    RunnableParallel(
        # Let the retriever fetch context based on the question
        context=RunnableLambda(lambda x: x["context"]),
        question=RunnablePassthrough(),
        history=RunnableLambda(lambda x: x.get("history", ""))
    )
    | RunnableLambda(lambda x: {
        "question": x["question"],
        "history": x["history"],
        "context": "\n".join([doc.page_content for doc in x["context"]])
    })
    | GENERAL_PROMPT
    | llm
    | StrOutputParser()
)

# Intent detection
def detect_intent(msg: str) -> str:
    product_words = ["lubricant", "oil", "grease", "engine", "product", "transmission"]
    return "product" if any(w in msg.lower() for w in product_words) else "general"

# Intent detection(general)
def detect_general_page_intent(query: str) -> str | None:
    query_lower = query.lower()
    for keyword in GENERAL_PAGES:
        if keyword in query_lower:
            return keyword
    return None

# Fuzzy matching
def fuzzy_match_product_by_name(query):
    docs = load_products_from_db()
    best_doc = None
    best_score = 0.0
    for doc in docs:
        name = doc.metadata.get("name", "").lower()
        score = SequenceMatcher(None, name, query.lower()).ratio()
        if score > best_score:
            best_doc = doc
            best_score = score
    return best_doc if best_score > 0.6 else None

def get_fresh_products():
    urls = crawl_product_pages()
    products = []
    for url in urls:
        prod = scrape_product_page(url)
        if prod:
            products.append(prod)
    return products

def match_query_to_product(query, products):
    best_score = 0
    best_product = None
    for prod in products:
        name = prod["name"].lower()
        score = SequenceMatcher(None, name, query.lower()).ratio()
        if score > best_score:
            best_score = score
            best_product = prod
    return best_product if best_score > 0.6 else None

# Main
def ask_bot(query: str, history: list[dict]):
    global last_product_doc
    from db import GENERAL_PAGES

    # intent = detect_intent(query)
    # Only take the last N history items (e.g., last 4 exchanges = 8 messages)
    N = 6
    recent_history = history[-N:] if len(history) > N else history

    formatted_history = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_history
    )

    # if intent == "product":
    #     if is_followup_question(query) and last_product_doc:
    #         context = last_product_doc["content"]
    #     else:
    #         products = get_fresh_products()
    #         matched = match_query_to_product(query, products)
    #         if not matched:
    #             return "I'm sorry, I couldn't find any matching product from our website."
    #         last_product_doc = matched
    #         context = matched["content"]

    #     return rag_chain_product.invoke({
    #         "question": query,
    #         "context": context,
    #         "history": formatted_history
    #     })

    # else:
    context = retriever.get_relevant_documents(query)

    response = rag_chain_general.invoke({
        "question": query,
        "context": context,
        "history": formatted_history
    })

    for i, doc in enumerate(context, 1):
        print(f"\n🔍 Document {i}:")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")

    response = re.sub(r"Best Regards,.*", "", response, flags=re.IGNORECASE).strip()

    query_lower = query.lower()
    matched_page = next((key for key in GENERAL_PAGES if key in query_lower), None)
    if matched_page:
        response += f"\n\nYou may also visit: {GENERAL_PAGES[matched_page]}"

    return response

def rebuild_vectorstore():
    global vectorstore, embedding, retriever

    print("[INFO] Rebuilding vectorstore...")

    documents = get_all_documents()
    print(f"[INFO] Loaded {len(documents)} documents before chunking.")

    chunks = chunk_documents(documents)
    print(f"[INFO] Created {len(chunks)} chunks after splitting.")

    try:
        if vectorstore:
            vectorstore.delete_collection()
            print("[INFO] Vectorstore collection cleared.")
    except Exception as e:
        print(f"[WARN] Failed to clear vectorstore: {e}")

    embedding = CohereEmbeddings(model="embed-multilingual-v3.0", cohere_api_key=COHERE_API_KEY)

    # Rebuild and save new vectorstore
    Chroma.from_documents(chunks, embedding, persist_directory=CHROMA_PATH).persist()

    print("[INFO] Vectorstore rebuilt and saved.")

    # Now reconnect cleanly to new vectorstore
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 7})

    print("[INFO] Reconnected to new vectorstore.")



