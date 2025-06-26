from live_scraper import crawl_product_pages, scrape_product_page
from bs4 import BeautifulSoup
import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

GENERAL_PAGES = {
    "about": "https://www.silvestreph.com/about",
    "contact": "https://www.silvestreph.com/contact",
    "faq": "https://www.silvestreph.com/help-center",
    "tracking": "https://www.silvestreph.com/tracking-page",
    "shipping and returns": "https://www.silvestreph.com/shipping-and-returns",
    "blog": "https://www.silvestreph.com/blogs",
    "partners": "https://www.silvestreph.com/blogs",
    "home": "https://www.silvestreph.com/"
}

def load_all_documents():
    documents = []

    # -- Load Product Pages --
    product_urls = crawl_product_pages()
    for url, category in product_urls:
        data = scrape_product_page(url, category)
        if data:
            doc = Document(
                page_content=data["description"],
                metadata={
                    "name": data["name"],
                    "url": data["url"],
                    "category": data["category"],
                    "type": "product",
                    "price": data["price"]
                }
            )
            documents.append(doc)

    # -- Load General Pages --
    headers = {"User-Agent": "Mozilla/5.0"}
    for label, url in GENERAL_PAGES.items():
        try:
            resp = requests.get(url, headers=headers, timeout=10)

            if resp.status_code == 404:
                print(f"[⚠️] Skipping 404 page: {url}")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            text = soup.get_text(separator="\n", strip=True)

            # 🧹 Skip pages that clearly contain product listings
            if "add to cart" in text.lower() and "price" in text.lower():
                print(f"[🧹] Skipping general page with embedded product listings: {url}")
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "url": url,
                        "type": "general",
                        "source": label
                    }
                )
            )

        except Exception as e:
            print(f"[WARN] Failed to load general page '{label}': {e}")

    # -- Chunking All Documents --
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunked_docs = splitter.split_documents(documents)

    print(f"✅ Loaded and chunked {len(chunked_docs)} total documents.")
    return chunked_docs


def get_all_product_names() -> list[str]:
    from sqlite3 import connect

    DB_PATH = "silvestre_products.db"
    conn = connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT name FROM products")
    results = [row[0] for row in cur.fetchall()]
    conn.close()
    return results
