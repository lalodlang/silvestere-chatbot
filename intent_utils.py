import re
from db import get_all_product_names

# Track follow-up state
last_intent = None
followup_count = 0
MAX_FOLLOWUPS = 3


def is_followup_question(query: str) -> bool:
    """
    Determines if the question sounds like a follow-up to a product discussion.
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
    url_match = re.search(r"https://www\.silvestreph\.com/product-page/([^)\s]+)", response)
    if url_match:
        return url_match.group(1).replace("-", " ").replace("%20", " ").strip()

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

    # Load product names
    product_names = get_all_product_names()

    # Priority 1: Exact product match
    for name in product_names:
        if name.lower() in text_lower:
            return "product"

    # Priority 2: Starts with "about" and a product name
    if text_lower.startswith("about"):
        after_about = text_lower.split("about", 1)[1].strip()
        for name in product_names:
            if name.lower() in after_about:
                return "product"
        return "about"

    # Priority 3: Product-like keywords
    product_keywords = [
        "engine oil", "lubricant", "gear oil", "grease", "synthetic",
        "tire", "transmission", "bentonite", "product", "oil"
    ]
    if any(kw in text_lower for kw in product_keywords):
        return "product"

    # Priority 4: General keyword mapping
    general_keywords = {
        "contact": ["contact", "how can i contact", "how do i contact", "get in touch", "phone", "email", "reach", "call", "location", "address"],
        "about": ["about", "mission", "vision", "company", "who are you"],
        "shipping": ["shipping", "delivery", "returns"],
        "warranty": ["warranty", "guarantee"],
        "terms": ["terms", "conditions"],
        "privacy": ["privacy", "data policy"],
        "faq": ["faq", "help", "common questions"],
        "tracking": ["track", "tracking", "order status"],
        "blog": ["blog", "news", "articles"],
        "partners": ["partners", "partnerships"],
        "home": ["home", "homepage"]
    }

    for intent_value, keywords in general_keywords.items():
        if any(k in text_lower for k in keywords):
            return intent_value

    return "general"




def update_followup_state(current_intent: str) -> bool:
    """
    Returns True if the user is within the follow-up window (≤3),
    and False if follow-up state should reset.
    """
    global last_intent, followup_count

    if current_intent == "product":
        if last_intent == "product":
            if followup_count < MAX_FOLLOWUPS:
                followup_count += 1
                print(f"[FOLLOW-UP COUNT] {followup_count}/3")
                return True
            else:
                print("[INFO] Max follow-ups reached. Resetting follow-up count.")
                followup_count = 0
                return False
        else:
            followup_count = 0
            last_intent = "product"
            return False
    else:
        followup_count = 0
        last_intent = current_intent
        return False
