import re
from db import get_all_product_names


def is_followup_question(query: str) -> bool:
    """
    Checks if the user's question appears to be a follow-up to a previously discussed product.
    """
    query_lower = query.lower()
    followup_keywords = [
        "how much", "price", "cost", "is it available in pails and drums", "does it", 
        "what size", "where", "can i", "do you offer", "does this product",
        "availability", "volume", "packaging", "what’s the", "do you have this"
    ]
    return any(keyword in query_lower for keyword in followup_keywords)


def extract_product_name(response: str) -> str:
    """
    Tries to extract a product name from a response string.
    First looks for a URL-based product name, then falls back to sentence parsing.
    """
    # Extract from product URL
    url_match = re.search(r"https://www\.silvestreph\.com/product-page/([^)\s]+)", response)
    if url_match:
        return url_match.group(1).replace("-", " ").replace("%20", " ").strip()

    # Fallback: extract from natural language
    fallback_match = re.search(
        r"(?:Product Name|product is|referring to is)[:\s]+([^\n.]+)", 
        response, 
        re.IGNORECASE
    )
    if fallback_match:
        return fallback_match.group(1).strip()

    return ""

def detect_intent(text: str) -> str:
    text_lower = text.lower().strip()

    # Known general pages
    general_keywords = {
        "about": "about",
        "contact": "contact",
        "privacy": "privacy",
        "terms": "terms",
        "shipping": "shipping",
        "warranty": "warranty"
    }

    # Load product names
    product_names = get_all_product_names()

    # Priority 1: Exact product match
    for name in product_names:
        if name.lower() in text_lower:
            return "product"

    # Priority 2: Starts with "about" → check next words
    if text_lower.startswith("about"):
        # Try to extract the rest after 'about'
        after_about = text_lower.split("about", 1)[1].strip()
        for name in product_names:
            if name.lower() in after_about:
                return "product"
        # Else fallback to general
        return "about"

    # Priority 3: Product-like keywords
    product_keywords = ["engine oil", "lubricant", "gear oil", "grease", "synthetic", "tire", "transmission", "bentonite", "product"]
    if any(kw in text_lower for kw in product_keywords):
        return "product"

    # Priority 4: General keywords
    for key, intent in general_keywords.items():
        if text_lower.startswith(key) or f" {key}" in text_lower:
            return intent

    # Default
    return "general"
