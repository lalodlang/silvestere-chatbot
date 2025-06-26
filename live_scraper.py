import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import hashlib

BASE_URL = "https://www.silvestreph.com"
START_URL = f"{BASE_URL}/shop"

def crawl_product_pages():
    CATEGORY_URLS = {
        "Industrial Lubricants": "Industrial%2520Lubricants",
        "Automotive Lubricants": "Automotive%2520Lubricants",
        "Marine Lubricants": "Marine%2520Lubricants",
        "Grease Lubricants": "Grease%2520Lubricants",
        "Specialty Lubricants": "Specialty%2520Lubricants",
        "Motorcycle Lubricants": "Motorcycle%2520Lubricants",
        "Motorcycle Tires": "Motorcycle%2520Tires"
    }

    headers = {"User-Agent": "Mozilla/5.0"}
    all_products = set()

    for category, encoded in CATEGORY_URLS.items():
        print(f"\n📂 Scraping category: {category}")
        for page in range(1, 10):
            url = f"{BASE_URL}/shop?Category={encoded}&page={page}"
            try:
                response = requests.get(url, headers=headers, timeout=10)
                soup = BeautifulSoup(response.text, "html.parser")

                links = {
                    urljoin(BASE_URL, a["href"])
                    for a in soup.find_all("a", href=True)
                    if "/product-page/" in a["href"]
                }

                if not links:
                    print(f"[!] No more products on page {page}, stopping.")
                    break

                for link in links:
                    all_products.add((link, category))

                print(f"[+] Page {page}: {len(links)} links")

            except Exception as e:
                print(f"[!] Failed to load {url}: {e}")

    print(f"\n✅ Total unique product URLs: {len(all_products)}")
    return list(all_products)

def compute_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def scrape_product_page(url: str, category: str = "Uncategorized") -> dict:
    try:
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        # Title
        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else "Untitled Product"

        # 💬 Extract from <pre data-hook="description"> directly
        description = ""
        pre_tag = soup.find("pre", attrs={"data-hook": "description"})
        if pre_tag:
            paragraphs = pre_tag.find_all("p")
            clean_paragraphs = [
                p.get_text(strip=True) for p in paragraphs
                if p.get_text(strip=True) and p.get_text(strip=True) != "\xa0"
            ]
            description = "\n\n".join(clean_paragraphs).strip()

        if not description:
            description = "No description available."

        price_tag = soup.find("span", {"data-hook": "formatted-primary-price"})
        price = price_tag.get_text(strip=True) if price_tag else "Contact us for pricing"

        availability = "Available in Pails and Drums"

        full_text = f"{title}\n\n{description}\n\nPrice: {price}\nAvailability: {availability}\nURL: {url}"

        return {
            "url": url,
            "name": title,
            "description": description + f"\n\nAvailability: {availability}",
            "price": price,
            "content": full_text,
            "hash": compute_hash(full_text),
            "category": category
        }

    except Exception as e:
        print(f"[ERROR] Failed to scrape ({url}, {category}): {e}")
        return None

if __name__ == "__main__":
    crawl_product_pages()
