# Silvestre Chatbot Assistant

A desktop-based intelligent assistant for Silvestre PH products, built with Python, CustomTkinter, LangChain, and RAG (Retrieval-Augmented Generation) architecture.

## ✨ Features

- 💬 Real-time product Q&A from (SilvestrePH) "https://www.silvestreph.com/"
- 🔎 Knowledge base scraping from sitemap + live pages
- 🧠 Memory and follow-up understanding
- 🌐 Clickable URLs in chat
- ♻️ Manual knowledge base refresh with spinner overlay
- 🖼️ CustomTkinter-based modern chat UI

---

## 📁 Project Structure
silvestere-chatbot/
│
├── assets/ # UI images like loading, spinner
├── db.py # Scraper + SQLite data loader
├── main.py # Main app launcher
├── ui.py # CustomTkinter UI layout and logic
├── rag_chain.py # LangChain RAG setup and bot response
├── refresh_and_rebuild.py # Utility to refresh the knowledge base
├── sitemap.xml # Sitemap for product discovery
├── requirements.txt # Python dependencies
├── .env # API keys (not pushed to GitHub)
└── README.md # You're here

```bash
git clone https://github.com/lalodlang/silvestere-chatbot.git
cd silvestere-chatbot

#Creating virtual Environment
python -m venv chatbot-env
.\chatbot-env\Scripts\activate  # On Windows

#Install Dependencies
pip install -r requirements.txt

#Run the app
python main.py

