# Silvestre Chatbot Assistant

A desktop-based intelligent assistant for Silvestre PH products, built with Python, CustomTkinter, LangChain, and RAG (Retrieval-Augmented Generation) architecture.

#Features

- Real-time product Q&A from (SilvestrePH) "https://www.silvestreph.com/"
- Knowledge base scraping from sitemap + live pages
- Memory and follow-up understanding
- Clickable URLs in chat
- Manual knowledge base refresh with spinner overlay
- CustomTkinter-based modern chat UI

---

Project Structure
silvestere-chatbot/
│
├── assets/ 
├── db.py 
├── main.py
├── ui.py 
├── rag_chain.py 
├── refresh_and_rebuild.py 
├── sitemap.xml 
├── requirements.txt 
├── .env 
└── README.md 

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

