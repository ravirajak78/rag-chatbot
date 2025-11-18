import streamlit as st
import os
import tempfile
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import docx
import google.generativeai as genai
from dotenv import load_dotenv

# ========== LOAD ENV ==========
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ========== CONFIG ==========
st.set_page_config(page_title="RAG Chatbot App", layout="wide")
DATA_DIR = Path(tempfile.gettempdir()) / "rag_uploads"
DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR = Path(__file__).resolve().parents[1] / "chroma_db"
COLLECTION_NAME = "rag_upload_collection"
EMBED_MODEL = "all-MiniLM-L6-v2"

# ========== STYLING ==========
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }

    .chat-wrapper {
        background: #fff;
        border-radius: 12px;
        padding: 15px;
        min-height: 65vh;
        max-height: 65vh;
        overflow-y: auto;
        box-shadow: 0 0 5px rgba(0,0,0,0.1);
        display: flex;
        flex-direction: column;
    }

    .chat-message {
        margin: 6px 0;
        padding: 10px 12px;
        border-radius: 10px;
        max-width: 85%;
        word-wrap: break-word;
        white-space: pre-wrap;
    }

    .user-msg {
        background: #d9e8ff;
        align-self: flex-end;
        text-align: right;
    }

    .bot-msg {
        background: #f2f4f7;
        align-self: flex-start;
    }

    .studio-card {
        background: #fff;
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 10px;
        box-shadow: 0 0 4px rgba(0,0,0,0.05);
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

# ========== LOADERS ==========
embedder = SentenceTransformer(EMBED_MODEL)
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_or_create_collection(COLLECTION_NAME)
model = genai.GenerativeModel("models/gemini-2.5-flash")

# ======================================
# SAFE GEMINI RESPONSE HANDLER
# ======================================
def safe_gemini_response(response):
    try:
        if hasattr(response, "candidates") and response.candidates:
            parts = response.candidates[0].content.parts
            if parts and hasattr(parts[0], "text"):
                txt = parts[0].text.strip()
                if txt:
                    return txt
        return "⚠️ The model returned an empty or blocked response."
    except:
        return "⚠️ Error: Gemini blocked or did not return a proper answer."

def generate_with_gemini(prompt, temperature=0.3):
    try:
        response = model.generate_content(
            prompt,
            generation_config={"temperature": temperature, "max_output_tokens": 1000}
        )
        return safe_gemini_response(response)
    except Exception as e:
        return f"⚠️ Gemini API error: {e}"

# ----------------------------------------
# DOCUMENT READERS
# ----------------------------------------
def read_pdf(file_path):
    text = ""
    reader = PdfReader(file_path)
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text

def read_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
    return splitter.split_text(text)

# ----------------------------------------
# SESSION STATE INIT
# ----------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []

if "active_doc" not in st.session_state:
    st.session_state.active_doc = None


# ---------- HEADER ----------
st.title("💬 RAG Chatbot App")
st.caption(" Built by *Ravi Rajak*")

# ---------- LAYOUT ----------
col_sources, col_chat, col_studio = st.columns([1.5, 2.5, 1.5])

# ================= LEFT PANEL ==================
with col_sources:
    st.header("📚 Upload & Sources")
    uploaded_files = st.file_uploader(
        "Upload up to 20 documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files[:20]:
            save_path = DATA_DIR / file.name

            with open(save_path, "wb") as f:
                f.write(file.read())

            if file.name not in st.session_state.uploaded_docs:
                st.session_state.uploaded_docs.append(file.name)

            ext = file.name.split(".")[-1]
            text = read_pdf(save_path) if ext == "pdf" else (
                read_docx(save_path) if ext == "docx" else read_txt(save_path)
            )

            chunks = chunk_text(text)
            embeddings = embedder.encode(chunks).tolist()
            ids = [f"{file.name}_{i}" for i in range(len(chunks))]

            collection.add(
                ids=ids,
                documents=chunks,
                metadatas=[{"source": file.name}] * len(chunks),
                embeddings=embeddings
            )

        st.success("✅ Files uploaded and indexed!")

    # Active document
    if st.session_state.uploaded_docs:
        st.session_state.active_doc = st.selectbox(
            "📄 Currently Working On:",
            st.session_state.uploaded_docs,
            index=0 if not st.session_state.active_doc else
            st.session_state.uploaded_docs.index(st.session_state.active_doc)
        )
    else:
        st.info("Upload documents to begin.")

# ================= CENTER PANEL ==================
with col_chat:
    st.header("💬 Chat")

    chat_container = st.empty()

    chat_html = "<div class='chat-wrapper' id='chat-box'>"
    for msg in st.session_state.messages:
        cls = "user-msg" if msg["role"] == "user" else "bot-msg"
        sender = "🧑‍💻 You" if msg["role"] == "user" else "🤖 Bot"
        chat_html += f"<div class='chat-message {cls}'><b>{sender}:</b><br>{msg['content']}</div>"
    chat_html += "</div>"

    chat_container.markdown(chat_html, unsafe_allow_html=True)

    # auto scroll
    st.markdown("""
        <script>
            const box = window.parent.document.querySelector('#chat-box');
            if (box) { box.scrollTop = box.scrollHeight; }
        </script>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Input + Clear button
    col_in, col_clear = st.columns([4, 1])

    with col_in:
        user_input = st.chat_input("Ask something...")

    with col_clear:
        if st.button("🧹 Clear"):
            st.session_state.messages = []
            st.rerun()

    # Chat processing
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        query_embedding = embedder.encode([user_input]).tolist()[0]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            where={"source": {"$eq": st.session_state.active_doc}}
        )

        retrieved = "\n\n".join(results["documents"][0])

        prompt = f"""
        Use ONLY the document context to answer.

        CONTEXT:
        {retrieved}

        QUESTION:
        {user_input}

        If answer not found, say: "Not found in document."
        """

        answer = generate_with_gemini(prompt)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

# ================= RIGHT PANEL ==================
with col_studio:
    st.header("🎛️ Studio Tools")

    if st.button("🧠 Summary"):
        txt = generate_with_gemini(
            f"Summarize the key points from {st.session_state.active_doc} in 6 bullet points."
        )
        st.markdown(f"<div class='studio-card'>{txt}</div>", unsafe_allow_html=True)

    if st.button("🗺️ Mind Map"):
        txt = generate_with_gemini(
            f"Create a mind-map hierarchy (bullet format) for {st.session_state.active_doc}."
        )
        st.markdown(f"<div class='studio-card'>{txt}</div>", unsafe_allow_html=True)

    if st.button("📊 Report"):
        txt = generate_with_gemini(
            f"Create a structured report (Intro, Key Points, Conclusion) for {st.session_state.active_doc}."
        )
        st.markdown(f"<div class='studio-card'>{txt}</div>", unsafe_allow_html=True)

    if st.button("❓ Quiz"):
        txt = generate_with_gemini(
            f"Generate 5 MCQ questions based on {st.session_state.active_doc} with correct answers marked."
        )
        st.markdown(f"<div class='studio-card'>{txt}</div>", unsafe_allow_html=True)
