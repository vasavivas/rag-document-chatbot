# 🤖 RAG Document Chatbot

Chat with any document using AI! Upload any `.txt` or `.pdf` file and ask questions in natural language. Built with LangChain, ChromaDB, and Groq's LLaMA 3.1.

---

## 🎯 What This Does

- **Chat with any document** — PDF or text file
- **AI answers from YOUR document** — not from general training data
- **Remembers conversation** — multi-turn chat with memory
- **Semantic search** — finds relevant content by meaning, not just keywords
- **Handles greetings** — natural conversation + document Q&A

---

## 🏗️ How It Works (RAG Pipeline)

```
Your Document
     ↓
TextLoader reads it
     ↓
TextSplitter cuts into chunks (500 chars each)
     ↓
HuggingFace converts chunks to embeddings (numbers)
     ↓
ChromaDB stores embeddings in vector database
     ↓
User asks question
     ↓
ChromaDB finds most relevant chunks
     ↓
LLaMA 3.1 answers based on found chunks ✅
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python** | Core language |
| **LangChain** | AI application framework |
| **Groq + LLaMA 3.1** | LLM for generating answers |
| **ChromaDB** | Vector database for semantic search |
| **HuggingFace Embeddings** | Converting text to vectors |
| **python-dotenv** | Secure API key management |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/vasavivas/rag-document-chatbot
cd rag-document-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup API key
```bash
# Copy example env file
cp .env.example .env

# Add your Groq API key (free at console.groq.com)
GROQ_API_KEY=your_key_here
```

### 4. Run the chatbot
```bash
python app.py
```

---

## 💬 Example Usage

```
🚀 RAG Document Chatbot Starting...

Enter document path: company_info.pdf

✅ Document loaded!
✅ Split into 24 chunks
✅ Vector database ready!

🤖 RAG DOCUMENT CHATBOT — Powered by LLaMA 3.1

You: hi
🤖 AI: Hello! I'm ready to answer questions about your document!

You: What are the main products?
🤖 AI: Based on the document, the main products are:
1. SmartBot - AI customer support chatbot
2. DataMind - Business intelligence platform
3. CodeAssist - AI code review tool

You: Who is the CEO?
🤖 AI: The CEO is Ravi Kumar, according to the document.

You: history
--- Conversation History ---
USER: hi
ASSISTANT: Hello! I'm ready...
USER: What are the main products?
ASSISTANT: Based on the document...
```

---

## 📁 Project Structure

```
rag-document-chatbot/
├── app.py              # Main application
├── requirements.txt    # Python dependencies
├── .env.example        # API key template
├── .gitignore          # Ignores .env and __pycache__
└── README.md           # This file
```

---

## 🔑 Key Concepts Used

**RAG (Retrieval Augmented Generation)**
Combines document retrieval with LLM generation for accurate, document-grounded answers.

**Vector Embeddings**
Text converted to numerical vectors so AI can search by semantic meaning.

**Conversation Memory**
Stores chat history so AI remembers context across multiple turns.

**Prompt Engineering**
Structured system prompts to keep AI answers consistent and document-focused.

---

## 👩‍💻 Author

**Vasavi Parla** — AI Engineer | React Developer
- LinkedIn: [vasavi-parla](https://www.linkedin.com/in/vasavi-parla-059a6aa0)
- 7+ years frontend experience, transitioning to AI Engineering

---

## 📄 License
MIT License — free to use and modify
