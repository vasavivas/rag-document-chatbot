"""
RAG Document Chatbot - Streamlit UI
====================================
Chat with any document using AI!
Built with LangChain + ChromaDB + Groq (LLaMA 3.1)

Author: Vasavi Parla
GitHub: https://github.com/vasavivas
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(page_title="RAG Document Chatbot", page_icon="🤖", layout="wide")

# ---------- Core RAG functions (same logic as original app.py) ----------

def load_document(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def split_into_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)


@st.cache_resource(show_spinner=False)
def create_vector_database(_chunks, file_key):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma.from_documents(documents=_chunks, embedding=embeddings)


def setup_ai():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant",
        temperature=0.1,
    )


PROMPT = ChatPromptTemplate.from_template(
    """
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

Answer:"""
)

GENERAL_CHAT_WORDS = [
    "hi", "hello", "hey", "how are you", "good morning", "good evening",
    "thanks", "thank you", "bye", "goodbye", "who are you", "what can you do",
]


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def format_history(history):
    if not history:
        return "No previous conversation"
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)


def get_answer(question, vectorstore, llm, history):
    is_general = any(word in question.lower() for word in GENERAL_CHAT_WORDS)
    if is_general:
        return llm.invoke(question).content
    relevant_docs = vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(question)
    chain = PROMPT | llm | StrOutputParser()
    return chain.invoke({
        "history": format_history(history),
        "context": format_docs(relevant_docs),
        "question": question,
    })


# ---------------------------- UI ----------------------------

st.title("🤖 RAG Document Chatbot")
st.caption("Chat with any document — powered by LangChain, ChromaDB & Groq LLaMA 3.1")

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(135deg, #f0f4ff 0%, #e8f0fe 100%);
        border: 2px dashed #6c8eff;
        border-radius: 12px;
        padding: 1.2rem 0.8rem;
    }
    section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"]:hover {
        border-color: #4361ee;
        background: linear-gradient(135deg, #e8f0fe 0%, #dde8ff 100%);
    }
    section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] {
        border-radius: 8px;
        border: 1px solid #4361ee;
        color: #4361ee;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 📄 Upload Document")
    st.caption("Drop a .txt or .pdf file to start chatting with it")
    uploaded_file = st.file_uploader(
        "Upload a .txt or .pdf file",
        type=["txt", "pdf"],
        label_visibility="collapsed",
    )

    if not os.getenv("GROQ_API_KEY"):
        st.warning("Set GROQ_API_KEY in your .env file to run this app.")

    st.markdown("---")
    st.markdown("**Tech Stack:** Python · LangChain · ChromaDB · HuggingFace Embeddings · Groq LLaMA 3.1")
    st.markdown("[GitHub Repo](https://github.com/vasavivas/rag-document-chatbot)")
    st.markdown("Built by **Vasavi Parla** | [LinkedIn](https://www.linkedin.com/in/vasavi-parla-059a6aa0)")

    if st.button("🔄 Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_file is not None:
    file_key = uploaded_file.name

    if st.session_state.get("current_file") != file_key:
        with st.spinner("📄 Processing document — loading, splitting, building vector database..."):
            suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            documents = load_document(tmp_path)
            chunks = split_into_chunks(documents)
            vectorstore = create_vector_database(chunks, file_key)
            llm = setup_ai()

            st.session_state.vectorstore = vectorstore
            st.session_state.llm = llm
            st.session_state.current_file = file_key
            st.session_state.messages = []

        st.success(f"✅ Document ready! Split into {len(chunks)} chunks.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about your document...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                answer = get_answer(
                    question,
                    st.session_state.vectorstore,
                    st.session_state.llm,
                    st.session_state.messages[:-1],
                )
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("👈 Upload a .txt or .pdf file from the sidebar to start chatting.")
    st.markdown(
        """
        **How it works:**
        1. Upload any document (.txt or .pdf)
        2. The document is split into chunks and converted into embeddings
        3. ChromaDB stores the embeddings for semantic search
        4. Ask questions — the AI finds relevant chunks and answers from your document
        """
    )
