import os
import re
import time
import random
from collections import defaultdict
from dotenv import load_dotenv
from typing import Optional
from langchain_core.documents import Document
from fuzzywuzzy import fuzz
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Chroma
from vectorstore_utils import load_vectorstore
from intent_utils import detect_intent, is_followup_question
from db import get_all_product_names, GENERAL_PAGES
from intent_utils import is_followup_question, update_followup_state
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)

# Load environment
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-70b-8192"
CHROMA_PATH = "chroma_db"

# Globals
last_product_doc: Optional[Document] = None

# Embedding + Vectorstore
embedding = CohereEmbeddings(
    model="embed-multilingual-v3.0", cohere_api_key=COHERE_API_KEY
)
vectorstore = load_vectorstore()

vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model=GROQ_MODEL,
    temperature=0.2
)

# PROMPTS
PRODUCT_PROMPT = PromptTemplate.from_template("""
You are a helpful and friendly assistant working for Silvestre Oil Company. 
Answer naturally and conversationally, as if you're speaking to a customer in real life. 
Use the context provided — if it’s missing or doesn’t help, let the user know politely.

<context>
{context}
</context>

<chat_history>
{history}
</chat_history>

User: {question}

Respond with a clear, human-sounding answer. Include product name, category, price, availability, and a link when relevant. Feel free to say something friendly or helpful at the end.
""")

GENERAL_PROMPT = PromptTemplate.from_template("""
You are a helpful assistant for Silvestre Oil Company. 
Speak politely and professionally, but sound like a real person.

Use the context below. If it's not relevant, kindly explain that.

<context>
{context}
</context>

<chat_history>
{history}
</chat_history>

User: {question}

Reply in a clear, customer-friendly way.
""")

FOLLOWUP_PROMPT = PromptTemplate.from_template("""
You are a helpful assistant for Silvestre Oil Company.

The customer is asking a **follow-up question** about this product:

<context>
{context}
</context>

<chat_history>
{history}
</chat_history>

User: {question}

Respond concisely and specifically to the follow-up question (e.g., just pricing, availability, packaging, etc.), without repeating full product info.
""")
  

# RAG CHAINS
rag_chain_product = (
    RunnableParallel(
        context=RunnableLambda(lambda x: x["context"]),
        question=RunnablePassthrough(),
        history=RunnableLambda(lambda x: x.get("history", ""))
    ) | PRODUCT_PROMPT | llm | StrOutputParser()
)

rag_chain_general = (
    RunnableParallel(
        context=RunnableLambda(lambda x: x["context"]),
        question=RunnablePassthrough(),
        history=RunnableLambda(lambda x: x.get("history", ""))
    ) | GENERAL_PROMPT | llm | StrOutputParser()
)

rag_chain_followup = (
    RunnableParallel(
        context=RunnableLambda(lambda x: x["context"]),
        question=RunnablePassthrough(),
        history=RunnableLambda(lambda x: x.get("history", ""))
    ) | FOLLOWUP_PROMPT | llm | StrOutputParser()
)



# MAIN BOT LOGIC
def normalize(text):
    return re.sub(r"[^\w\s]", "", text.lower().strip())

def clean_price_fallback_lines(response: str, price: str) -> str:
    """Remove default fallback pricing sentences if a valid price exists."""
    if price.lower() in ["contact us for pricing", "n/a", "not available"]:
        return response 

def ask_bot(query: str, history: list[dict]) -> str:
    global last_product_doc
    matched = None
    matched_from_semantic = False

    intent = detect_intent(query)
    if is_followup_question(query):
        intent = "product"
        
    formatted_history = "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in history[-3:])

    # Reset memory if switching away from product
    if intent != "product" and last_product_doc:
        print("[INFO] Switching away from product intent. Resetting memory.")
        last_product_doc = None

    # Shortcut: List products
    if any(keyword in query.lower() for keyword in [
        "what are your products", "list your products",
        "show me your products", "can i see your products",
        "what products do you have"]):
        return get_all_products()

    # --------------------- PRODUCT INTENT ---------------------
    if intent == "product":
        is_followup = is_followup_question(query) or update_followup_state(intent)
        if is_followup:
            if not last_product_doc:
                return "Please mention a specific product so I can assist you better."
            matched = last_product_doc
            print(f"[FOLLOW-UP] Reusing last product: {matched.metadata.get('name')}")
        else:
            # Step 1: Retriever-based match
            docs = retriever.get_relevant_documents(query)
            best_doc = None
            best_score = 0

            for d in docs:
                if d.metadata.get("type") != "product":
                    continue
                name = d.metadata.get("name", "").lower()
                score = fuzz.token_set_ratio(query.lower(), name)
                print(f"[RETRIEVER SEMANTIC SCORE] {score:.2f} for '{name}'")
                if score > best_score:
                    best_doc = d
                    best_score = score

            if best_doc and best_score >= 88:
                matched = best_doc
                matched_from_semantic = True
                print(f"[RETRIEVER MATCH] Matched: {matched.metadata.get('name')} (Score: {best_score})")

            # Step 2: Fuzzy fallback
            if not matched:
                print("[INFO] No strong vector match. Trying fuzzy fallback.")
                all_meta = vectorstore._collection.get(include=["metadatas"])["metadatas"]
                product_meta = [m for m in all_meta if m.get("type") == "product"]

                def normalize(text): return re.sub(r"[^\w\s]", "", text.lower())

                best_match = None
                highest_score = 0

                for meta in product_meta:
                    name = meta.get("name", "")
                    if not name:
                        continue
                    score = fuzz.token_set_ratio(normalize(query), normalize(name))
                    print(f"[DEBUG] Fuzzy score: {score:.2f} | '{name}'")
                    if score > highest_score:
                        best_match = meta
                        highest_score = score

                if best_match and (highest_score >= 80 or matched_from_semantic):
                    print(f"[MATCH FOUND] Accepting fuzzy match: {best_match['name']} (Score: {highest_score})")
                    matched = Document(page_content="No description available.", metadata=best_match)
                else:
                    print(f"[NO MATCH] Best fuzzy score: {highest_score}")
                    return "I couldn’t find a product with that name. Please try rephrasing it."

            last_product_doc = matched
            print(f"[NEW PRODUCT] Found: {matched.metadata.get('name')}")

        # --------- Build context and run product RAG chain ---------
        name = matched.metadata.get("name", "this product")
        url = matched.metadata.get("url", "")
        category = matched.metadata.get("category", "Uncategorized")
        price = matched.metadata.get("price", "Contact us for pricing")

        context = f"""Product Name: {name}
Category: {category}
Availability: Available in Pails and Drums
URL: {url}

Description:
{matched.page_content}
"""

        # Retry mechanism
        response = ""
        for attempt in range(3):
            try:
                chain = rag_chain_followup if is_followup else rag_chain_product
                response = chain.invoke({
                    "question": query,
                    "context": context,
                    "history": formatted_history
                })
                if response:
                    response = response.strip()
                break
            except Exception as e:
                print(f"[RETRY {attempt + 1}] Product query failed:", e)
                time.sleep(1)

        if not response:
            return "Sorry, I couldn’t process your product question right now. Please try again later."

        # Clean price fallback if real price is available
        if price != "Contact us for pricing":
            patterns = [
                r"(?i)as for pricing.*?\.",
                r"(?i)price\s*[:\-–]?\s*(not available|n/a|unknown).*?(\n|$)",
                r"(?i)unfortunately.*?(price|pricing).*?\.",
                r"(?i)pricing.*?not.*?(available|provided).*?\.",
                r"(?i)i don’t have.*?(price|pricing).*?\."
            ]
            for p in patterns:
                response = re.sub(p, "", response).strip()

        # Final formatting
        response += f"\n\nPrice: {price}"
        response += f"\nCategory: {category}"
        if url:
            response += f"\nProduct Page: {url}"

        return response

    # --------------------- GENERAL INTENT ---------------------
    try:
        about_keywords = ["journey", "growth", "promise", "mission", "vision", "offer", "beginnings"]
        query_lower = query.lower()

        if any(k in query_lower for k in about_keywords):
            print("[INFO] Keyword matches about page intent.")
            raw = vectorstore._collection.get(include=["documents", "metadatas"])
            about_docs = [
                doc for doc, meta in zip(raw["documents"], raw["metadatas"])
                if meta.get("source") == "about" or "about" in meta.get("url", "")
            ]
            context = "\n".join(about_docs)[:5000]
            response = rag_chain_general.invoke({
                "question": query,
                "context": context,
                "history": formatted_history
            }).strip()
            response += "\n\nLearn more: https://www.silvestreph.com/about"
            return response

        docs = retriever.get_relevant_documents(query)
        context_docs = [d for d in docs if d.metadata.get("type") != "product"]

        relevance_scores = [
            fuzz.token_set_ratio(query.lower(), d.page_content.lower()) for d in context_docs
        ]
        max_score = max(relevance_scores) if relevance_scores else 0

        if not context_docs or max_score < 50:
            print(f"[FILTER] Ignored query '{query}' due to low relevance (score: {max_score})")
            return "Sorry, I couldn’t find information related to your question."

        context = "\n".join(d.page_content for d in context_docs)[:5000]
        response = rag_chain_general.invoke({
            "question": query,
            "context": context,
            "history": formatted_history
        }).strip()

        if intent in GENERAL_PAGES:
            response += f"\n\nYou may also visit: {GENERAL_PAGES[intent]}"
        return response

    except Exception as e:
        print("[ERROR] General query failed:", e)
        return "Sorry, I couldn’t process your request right now."


# PRODUCT LISTING
def get_all_products():
    """Returns 3–5 random products per category from the vectorstore."""
    try:
        raw = vectorstore._collection.get(include=["metadatas"])
        product_meta = [m for m in raw["metadatas"] if m.get("type") == "product"]

        categories = defaultdict(list)
        for meta in product_meta:
            cat = meta.get("category", "Uncategorized")
            categories[cat].append(meta)

        response_lines = ["Here’s a sample of our products categorized for your convenience:\n"]

        for cat, products in categories.items():
            sampled = random.sample(products, min(len(products), 5))
            response_lines.append(f"{cat.upper()}:")
            for p in sampled:
                name = p.get("name", "Unnamed Product")
                url = p.get("url", "")
                line = f"- {name}" + (f"{url})" if url else "")
                response_lines.append(line)
            response_lines.append("")  

        return "\n".join(response_lines)

    except Exception as e:
        print("[ERROR] Failed to fetch products from vectorstore:", e)
        return "Sorry, I couldn’t fetch the product list at the moment."

def reload_vectorstore():
    global vectorstore, retriever
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    print("[INFO] Vectorstore reloaded in memory.")

