"""
RAG Document Chatbot
====================
Chat with any document using AI!
Built with LangChain + ChromaDB + Groq (LLaMA 3.1)

Author: Vasavi Parla
GitHub: https://github.com/vasavivas
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load API keys from .env file
load_dotenv()



# STEP 1: Load the document
# Supports both .txt and .pdf files
def load_document(file_path):
    print(f"\n📄 Loading document: {file_path}")

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")

    documents = loader.load()
    print(f"✅ Loaded {len(documents)} page(s)")
    return documents


# STEP 2: Split document into small chunks
# AI can't process huge documents at once
# So we split into smaller pieces

def split_into_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # each chunk = 500 characters
        chunk_overlap=50     # 50 char overlap so context isn't lost between chunks
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks")
    return chunks



# STEP 3: Create Vector Database
# Converts text chunks into numbers (embeddings)
# Stores them so we can search by meaning

def create_vector_database(chunks):
    print("Creating vector database (first run takes ~2 min)...")

    # HuggingFace converts text to numbers AI can search
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Chroma stores and searches those numbers
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    print("✅ Vector database ready!")
    return vectorstore



# STEP 4: Setup AI Model
# Using Groq's LLaMA 3.1 - fast and free!

def setup_ai():
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant",
        temperature=0.1  # low = consistent factual answers
    )
    return llm



# STEP 5: Build RAG Chain
# Connects: Question → Search DB → Send to AI → Answer

def build_rag_chain(vectorstore, llm):

    # This prompt tells AI to ONLY answer from document
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions based on the provided document.

Rules:
- Answer ONLY from the document context below
- If the answer is not in the document, say "I couldn't find that in the document"
- Be clear and concise
- If asked general questions or greetings, respond naturally

Conversation History:
{history}

Document Context:
{context}

Question: {question}

Answer:""")

    # Helper: joins all found document chunks into one text
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Helper: formats conversation history for AI
    def format_history(history):
        if not history:
            return "No previous conversation"
        return "\n".join(
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in history
        )

    return prompt, format_docs, format_history



# STEP 6: Main Chat Loop
# Takes user questions, searches document,
# sends to AI, returns answer

def start_chat(vectorstore, llm, prompt, format_docs, format_history):

    # Parse AI output to clean string
    output_parser = StrOutputParser()

    # Words that mean it's a general chat (not document question)
    general_chat_words = [
        "hi", "hello", "hey", "how are you",
        "good morning", "good evening", "thanks",
        "thank you", "bye", "goodbye", "who are you",
        "what can you do"
    ]

    # Memory: stores full conversation
    conversation_history = []

    print("\n" + "=" * 55)
    print("🤖  RAG DOCUMENT CHATBOT  —  Powered by LLaMA 3.1")
    print("=" * 55)
    print("💬  Ask anything about your document!")
    print("📋  Type 'history'  → see conversation")
    print("🔄  Type 'clear'    → reset memory")
    print("❌  Type 'exit'     → quit\n")

    while True:
        question = input("You: ").strip()

        # Handle special commands
        if not question:
            continue

        if question.lower() == "exit":
            print("\n👋 Goodbye! Happy learning!")
            break

        if question.lower() == "history":
            if not conversation_history:
                print("📭 No history yet!\n")
            else:
                print("\n--- Conversation History ---")
                for msg in conversation_history:
                    print(f"{msg['role'].upper()}: {msg['content']}")
                print("----------------------------\n")
            continue

        if question.lower() == "clear":
            conversation_history = []
            print("✅ Memory cleared!\n")
            continue

        # Save user question to memory
        conversation_history.append({
            "role": "user",
            "content": question
        })

        # Decide: general chat OR document question?
        is_general = any(
            word in question.lower()
            for word in general_chat_words
        )

        print("🤔 Thinking...\n")

        if is_general:
            # General chat — no document search needed
            response = llm.invoke(question)
            answer = response.content
        else:
            # Document question — search + AI answer
            # 1. Search vector DB for relevant chunks
            relevant_docs = vectorstore.as_retriever(
                search_kwargs={"k": 3}  # get top 3 relevant chunks
            ).invoke(question)

            # 2. Build the chain and get answer
            chain = prompt | llm | output_parser
            answer = chain.invoke({
                "history": format_history(conversation_history[:-1]),
                "context": format_docs(relevant_docs),
                "question": question
            })

        # Save AI answer to memory
        conversation_history.append({
            "role": "assistant",
            "content": answer
        })

        print(f"🤖 AI: {answer}\n")
        print("-" * 45 + "\n")



# MAIN — Entry point of the app

def main():
    print("\n🚀 RAG Document Chatbot Starting...")
    print("=" * 55)

    # Ask user which file to chat with
    print("\nSupported formats: .txt, .pdf")
    file_path = input("Enter document path (or press Enter for demo): ").strip()

    # Use demo file if nothing entered
    if not file_path:
        file_path = "demo_document.txt"
        # Create demo file if it doesn't exist
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write("""
Company: TechCorp AI Solutions
Founded: 2020
Location: Hyderabad, India

Products:
- SmartBot: AI customer support chatbot
- DataMind: Business intelligence platform
- CodeAssist: AI code review tool

Team Size: 150 employees

Technologies Used:
Python, React, LangChain, OpenAI, AWS

Revenue: $5M ARR in 2024

CEO: Ravi Kumar
CTO: Priya Sharma
""")
            print("✅ Demo document created!")

    # Run all steps
    documents = load_document(file_path)
    chunks = split_into_chunks(documents)
    vectorstore = create_vector_database(chunks)
    llm = setup_ai()
    prompt, format_docs, format_history = build_rag_chain(vectorstore, llm)

    # Start chatting!
    start_chat(vectorstore, llm, prompt, format_docs, format_history)


if __name__ == "__main__":
    main()
