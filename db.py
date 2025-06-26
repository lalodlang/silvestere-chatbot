# db.py
import os
import shutil
from live_scraper import crawl_product_pages, scrape_product_page
import sqlite3
from bs4 import BeautifulSoup
import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from urllib.parse import urljoin
import re


DB_PATH = "silvestre_products.db"
CHROMA_PATH = "chroma_db"
AVAILABILITY = "Available in Pails and Drums"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

GENERAL_PAGES = {
    "about": "https://www.silvestreph.com/about",
    "contact": "https://www.silvestreph.com/contact",
    "partners": "https://www.silvestreph.com/partners",
    "ship": "https://www.silvestreph.com/shipping-and-returns",
    "shop": "https://www.silvestreph.com/shop"
}

# "track": "https://www.silvestreph.com/tracking-page", removed this since most information in the url is already given in "contact"

# Chunking
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    return chunks

# --- Scrape General Pages ---
def scrape_general_pages():
    docs = []

    # Scrape static general pages
    for title, url in GENERAL_PAGES.items():
        try:
            res = requests.get(url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(res.content, "html.parser")

            body = soup.find("main") or soup.find("body") or soup
            text = body.get_text(separator="\n", strip=True)
            cleaned = "\n".join([line.strip() for line in text.splitlines() if line.strip()])

            if cleaned:
                docs.append(Document(
                    page_content=f"{title.upper()}\n{cleaned}\n{url}",
                    metadata={
                        "title": title,
                        "url": url,
                        "type": title.lower(),
                        "tags": title.lower()
                    }
                ))
        except Exception as e:
            print(f"[ERROR] General page {url}: {e}")

    # Scrape paginated shop pages
    shop_base_url = "https://www.silvestreph.com/shop?page="
    page_num = 2

    while True:
        try:
            shop_url = f"{shop_base_url}{page_num}"
            print(f"[INFO] Scraping shop page {page_num}: {shop_url}")
            res = requests.get(shop_url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(res.content, "html.parser")

            # Extract product cards
            product_blocks = []
            for card in soup.select('a[data-hook="product-item-container"]'):
                href = urljoin(shop_url, card.get("href", ""))

                container = card.find_parent()
                name = container.get("aria-label", "").strip() if container else "Unnamed Product"

                price_el = container.find_next(string=lambda t: "₱" in t) if container else None
                price = price_el.strip() if price_el else "Price not found"

                block = f"{name}\nPrice\n{price}\nAdd to Cart\n{href}"
                product_blocks.append(block)

            # Join all product blocks (no unrelated content mixed in)
            product_content = "\n\n".join(product_blocks) if product_blocks else "No products found."

            # Stop when no more products
            if product_content == "No products found.":
                print(f"[INFO] Reached end of shop pages at page {page_num}.")
                break

            # Build the document
            page_content = (
                f"This page contains the list of products in page {page_num}\n"
                f"{product_content}\n"
                f"{shop_url}"
            )

            docs.append(Document(
                page_content=page_content,
                metadata={
                    "title": f"shop-page-{page_num}",
                    "url": shop_url,
                    "type": "shop",
                    "tags": "shop"
                }
            ))

            page_num += 1

        except Exception as e:
            print(f"[ERROR] Shop page {shop_url}: {e}")
            break

    return docs

# --- Refresh SQLite DB from Live Crawl ---
def refresh_database():
    # Try live crawl first
    print("[INFO] Crawling live website for product pages...")
    product_urls = crawl_product_pages()
    print(f"[INFO] Found {len(product_urls)} product URLs from live crawl")

    # Fallback to sitemap if too few
    if len(product_urls) < 100:
        print("[INFO] Low product count from live crawl. Falling back to sitemap...")
        product_urls = crawl_product_pages()
        print(f"[INFO] Found {len(product_urls)} product URLs from live crawl")
        print(f"[INFO] Found {len(product_urls)} product URLs from sitemap")

    # Refresh DB
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("DROP TABLE IF EXISTS products")
    c.execute("""
        CREATE TABLE products (
            name TEXT,
            description TEXT,
            price TEXT,
            url TEXT
        )
    """)

    inserted = 0
    for url in product_urls:
        data = scrape_product_page(url)
        if data:
            c.execute("INSERT INTO products VALUES (?, ?, ?, ?)", (data["name"], data["description"], data["price"], data["url"]))
            inserted += 1

    conn.commit()
    conn.close()
    print(f"✅ Inserted {inserted} products into {DB_PATH}")

# --- Load for Vectorstore ---
def load_products_from_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, description, price, url FROM products")
    rows = c.fetchall()
    conn.close()

    documents = []
    for name, desc, price, url in rows:
        metadata = {
            "name": name,
            "price": price,
            "url": url,
            "type": "product",
            "tags": "product"
        }
        content = f"Product Name: {name}\nPrice: {price}\nDescription: {desc}\nURL: {url}"
        documents.append(Document(page_content=content, metadata=metadata))

    documents += scrape_general_pages()
    return documents

def get_all_documents():
    general_docs = scrape_general_pages()

    return general_docs

# --- Manual Trigger ---
if __name__ == "__main__":
    refresh_database()
