
from dotenv import load_dotenv
import sys
from ui import ChatApp

# Load environment variables
load_dotenv()

# Log info
print("[INFO] Embedding Model: Cohere (cloud-based)")
print("[INFO] Language Model: Groq LLaMA3-70B (cloud-based)")

# Entry point
if __name__ == "__main__":
    try:
        app = ChatApp()
        app.mainloop()
    except Exception as e:
        print(f"[ERROR] Failed to start application: {e}")
        sys.exit(1)
