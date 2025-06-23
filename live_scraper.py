# live_scraper.py

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import hashlib

BASE_URL = "https://www.silvestreph.com"
START_URL = f"{BASE_URL}/shop"

def crawl_product_pages(start_url=START_URL, max_depth=2):
    visited = set()
    to_visit = [(start_url, 0)]
    product_urls = set()

    while to_visit:
        current_url, depth = to_visit.pop(0)
        if current_url in visited or depth > max_depth:
            continue

        visited.add(current_url)
        try:
            res = requests.get(current_url, timeout=10)
            soup = BeautifulSoup(res.text, "html.parser")
        except Exception:
            continue

        # Only keep exact matches for /product-page/
        if "/product-page/" in current_url:
            product_urls.add(current_url)

        for link in soup.find_all("a", href=True):
            href = link["href"]
            full_url = urljoin(current_url, href)
            if urlparse(full_url).netloc == urlparse(BASE_URL).netloc and full_url not in visited:
                to_visit.append((full_url, depth + 1))

    return list(product_urls)

def compute_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def scrape_product_page(url: str) -> dict:
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        # Title
        title = soup.find("h1").get_text(strip=True) if soup.find("h1") else "Untitled Product"

        # Description
        desc = soup.find("div", class_="elementor-widget-theme-post-content")
        description = desc.get_text(separator="\n", strip=True) if desc else "No description available."

        # Price (real value from product page)
        price_tag = soup.find("span", {"data-hook": "formatted-primary-price"})
        price = price_tag.get_text(strip=True) if price_tag else "Contact us for pricing"

        # Availability
        availability = "Available in Pails and Drums"

        # Combine into full content block
        full_text = f"{title}\n\n{description}\n\nPrice: {price}\nAvailability: {availability}\nURL: {url}"

        return {
            "url": url,
            "name": title,
            "description": description + f"\n\nAvailability: {availability}",
            "price": price,
            "content": full_text,
            "hash": compute_hash(full_text),
        }

    except Exception as e:
        print(f"[ERROR] Failed to scrape {url}: {e}")
        return None

