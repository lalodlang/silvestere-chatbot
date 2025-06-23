# intent_utils.pyy

def is_followup_question(query: str) -> bool:
    followup_keywords = [
        "how much", "price", "cost", "is it available", "does it", 
        "what size", "where", "can i", "do you offer", "does this product"
    ]
    return any(kw in query.lower() for kw in followup_keywords)


def extract_product_name(response: str) -> str:
    import re
    # Try extracting from the product URL if present
    match = re.search(r"https://www\.silvestreph\.com/product-page/([^)\s]+)", response)
    if match:
        return match.group(1).replace("-", " ").replace("%20", " ")
    
    # Fallback: try to extract from sentence
    name_match = re.search(r"(Product Name|product is|referring to is)[:\s]+([^\n.]+)", response, re.IGNORECASE)
    if name_match:
        return name_match.group(2).strip()
    
    return ""
