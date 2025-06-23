# db.py
from live_scraper import crawl_product_pages, scrape_product_page
import sqlite3
from bs4 import BeautifulSoup
import requests
from langchain_core.documents import Document


DB_PATH = "silvestre_products.db"
AVAILABILITY = "Available in Pails and Drums"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

GENERAL_PAGES = {
    "about": "https://www.silvestreph.com/about",
    "contact": "https://www.silvestreph.com/contact",
    "faq": "https://www.silvestreph.com/help-center",
    "partners": "https://www.silvestreph.com/partners",
    "ship": "https://www.silvestreph.com/shipping-and-returns",
    "track": "https://www.silvestreph.com/tracking-page"
}

# --- Scrape General Pages ---
def scrape_general_pages():
    docs = []
    for title, url in GENERAL_PAGES.items():
        try:
            res = requests.get(url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(res.content, "html.parser")

            body = soup.find("main") or soup.find("body") or soup
            text = body.get_text(separator="\n", strip=True)
            cleaned = "\n".join([line.strip() for line in text.splitlines() if line.strip()])

            if not cleaned:
                continue

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

# --- Manual Trigger ---
if __name__ == "__main__":
    refresh_database()
