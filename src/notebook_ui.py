import streamlit as st
import os
import tempfile
from pathlib import Path
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import docx
import google.generativeai as genai
from google.generativeai import embed_content
from dotenv import load_dotenv
from typing import Optional

# ========== LOAD ENV ==========
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ========== CONFIG ==========
st.set_page_config(page_title="RAG Chatbot App", layout="wide")
DATA_DIR = Path(tempfile.gettempdir()) / "rag_uploads"
DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR = Path(__file__).resolve().parents[1] / "chroma_db"
COLLECTION_NAME = "rag_chatbot_collection"

# ========== STYLING ==========
st.markdown(
    """
<style>
    .block-container { padding-top: 1rem; }

    /* chat wrapper (center column) */
    .chat-wrapper {
        background: #fff;
        border-radius: 12px;
        padding: 15px;
        min-height: 62vh;
        max-height: 62vh;
        overflow-y: auto;
        box-shadow: 0 0 8px rgba(0,0,0,0.08);
        display: flex;
        flex-direction: column;
        gap: 6px;
    }

    .chat-message {
        margin: 6px 0;
        padding: 10px 12px;
        border-radius: 10px;
        max-width: 85%;
        word-wrap: break-word;
        white-space: pre-wrap;
        line-height: 1.4;
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
        box-shadow: 0 0 6px rgba(0,0,0,0.04);
        white-space: pre-wrap;
    }

    /* visual container to center the chat input under the center column */
    .centered-input-wrapper {
        max-width: 900px;  /* adjust to match center column visual width */
        margin: 8px auto 36px auto; /* center and give bottom spacing */
        display: flex;
        gap: 8px;
        align-items: center;
    }

    /* style the "Clear" button area next to the input by using Streamlit's default classes */
</style>
""",
    unsafe_allow_html=True,
)

# ========== CHROMA DB ==========
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_or_create_collection(COLLECTION_NAME)

# ========== GEMINI MODEL ==========
model = genai.GenerativeModel("models/gemini-2.5-flash")


# -------- EMBEDDING FUNCTION --------
def get_embedding(text: str) -> list:
    """
    Use Gemini embeddings via google.generativeai.embed_content.
    Returns the embedding vector (list of floats).
    """
    try:
        resp = embed_content(model="models/text-embedding-004", content=text)
        return resp["embedding"]
    except Exception as e:
        # propagate or return a placeholder — here we raise so developer can see logs
        raise RuntimeError(f"Embedding error: {e}")


# -------- GEMINI SAFE RESPONSE HANDLER --------
def safe_gemini_response(response) -> str:
    """
    Extract text safely from various Gemini response shapes.
    """
    try:
        # new SDK quick access
        if hasattr(response, "text") and response.text:
            return response.text.strip()

        # fallback to candidates/parts
        if hasattr(response, "candidates") and response.candidates:
            cand = response.candidates[0]
            if getattr(cand, "content", None) and getattr(cand.content, "parts", None):
                part = cand.content.parts[0]
                return getattr(part, "text", "").strip() or "⚠️ Empty part text."

        return "⚠️ The model returned no usable text (possibly blocked)."
    except Exception as e:
        return f"⚠️ Error extracting model response: {e}"


def generate_with_gemini(prompt: str, temperature: float = 0.2) -> str:
    """
    Call Gemini generate_content and return safe text.
    """
    try:
        response = model.generate_content(
            prompt,
            generation_config={"temperature": temperature, "max_output_tokens": 900},
        )
        return safe_gemini_response(response)
    except Exception as e:
        return f"⚠️ Gemini API error: {e}"


# -------- DOCUMENT READERS --------
def read_pdf(file_path: Path) -> str:
    text = ""
    reader = PdfReader(str(file_path))
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text


def read_docx(file_path: Path) -> str:
    doc = docx.Document(str(file_path))
    return "\n".join(p.text for p in doc.paragraphs)


def read_txt(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def chunk_text(text: str) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
    return splitter.split_text(text)


# -------- SESSION STATE INIT --------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": "..."}
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []  # list of filenames
if "active_doc" not in st.session_state:
    st.session_state.active_doc = None


# ---------- HEADER ----------
st.title("💬 RAG Chatbot App")
st.caption("Built by *Ravi Rajak* | NotebookLM-style RAG")

# ---------- LAYOUT ----------
col_sources, col_chat, col_studio = st.columns([1.5, 2.5, 1.5])

# ================= LEFT PANEL ==================
with col_sources:
    st.header("📚 Upload & Sources")
    uploaded_files = st.file_uploader(
        "Upload up to 20 documents (pdf/docx/txt)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        # iterate and index each file
        for file in uploaded_files[:20]:
            save_path = DATA_DIR / file.name
            with open(save_path, "wb") as f:
                f.write(file.read())

            if file.name not in st.session_state.uploaded_docs:
                st.session_state.uploaded_docs.append(file.name)

            ext = file.name.split(".")[-1].lower()
            text = (
                read_pdf(save_path)
                if ext == "pdf"
                else read_docx(save_path)
                if ext == "docx"
                else read_txt(save_path)
            )

            # chunk + embed (note: embedding each chunk via API)
            chunks = chunk_text(text)
            embeddings = []
            for chunk in chunks:
                try:
                    embeddings.append(get_embedding(chunk))
                except Exception as e:
                    st.error(f"Embedding failed for chunk: {e}")
                    embeddings.append([0.0])  # placeholder to keep lengths consistent

            ids = [f"{file.name}_{i}" for i in range(len(chunks))]
            metadatas = [{"source": file.name, "chunk_index": i} for i in range(len(chunks))]

            collection.add(
                ids=ids,
                documents=chunks,
                metadatas=metadatas,
                embeddings=embeddings,
            )

        st.success("✅ Files uploaded & indexed!")

    # Show/select uploaded docs
    if st.session_state.uploaded_docs:
        st.session_state.active_doc = st.selectbox(
            "📄 Currently Working On",
            st.session_state.uploaded_docs,
            index=0
            if st.session_state.active_doc is None
            else st.session_state.uploaded_docs.index(st.session_state.active_doc),
        )
    else:
        st.info("Upload documents to get started.")


# ================= CENTER PANEL ==================
with col_chat:
    st.header("💬 Chat")

    # Render chat messages inside a scrollable div (chat-wrapper)
    chat_box = st.empty()

    html = "<div class='chat-wrapper' id='chat-box'>"
    for msg in st.session_state.messages:
        cls = "user-msg" if msg["role"] == "user" else "bot-msg"
        sender = "🧑‍💻 You" if msg["role"] == "user" else "🤖 Bot"
        # keep message content safe-escaped by Streamlit's markdown, using raw HTML with prewrap
        html += f"<div class='chat-message {cls}'><b>{sender}:</b><br>{st.markdown(msg['content'], unsafe_allow_html=False) or msg['content']}</div>"
    html += "</div>"

    # As Streamlit's markdown escapes content, we will render raw container but messages already safe
    # To avoid double-escaping, re-build using simple HTML for known safe messages:
    html = "<div class='chat-wrapper' id='chat-box'>"
    for msg in st.session_state.messages:
        cls = "user-msg" if msg["role"] == "user" else "bot-msg"
        sender = "🧑‍💻 You" if msg["role"] == "user" else "🤖 Bot"
        content = msg["content"].replace("\n", "<br>")
        html += f"<div class='chat-message {cls}'><b>{sender}:</b><br>{content}</div>"
    html += "</div>"

    chat_box.markdown(html, unsafe_allow_html=True)

    # Auto-scroll (tries to scroll the wrapper to bottom)
    st.markdown(
        """
    <script>
        (function() {
            const parent = window.parent.document;
            const box = parent.querySelector('#chat-box');
            if (box) { box.scrollTop = box.scrollHeight; }
        })()
    </script>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

# ------------------ Chat input (ROOT LEVEL) ------------------
# Place st.chat_input() OUTSIDE columns but visually center it under center column.
st.markdown(
    "<div class='centered-input-wrapper'>", unsafe_allow_html=True
)

# IMPORTANT: st.chat_input must NOT be inside any with col_... block
user_input = st.chat_input("Ask a question...")

# Add Clear button to the right visually using a small column trick afterwards
col_for_clear = st.columns([1, 1, 1, 1, 1])[4]  # create some spacing columns, pick last for clear
with col_for_clear:
    if st.button("🧹 Clear"):
        st.session_state.messages = []
        st.experimental_rerun()

st.markdown("</div>", unsafe_allow_html=True)

# ------------------ Process input ------------------
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    try:
        query_vector = get_embedding(user_input)
    except Exception as e:
        st.error(f"Failed to create query embedding: {e}")
        query_vector = None

    # prepare query; if active_doc selected, filter by source
    where_clause: Optional[dict] = None
    if st.session_state.active_doc:
        where_clause = {"source": {"$eq": st.session_state.active_doc}}

    # Query Chroma
    try:
        if query_vector is not None:
            results = (
                collection.query(
                    query_embeddings=[query_vector],
                    n_results=5,
                    where=where_clause,
                )
                if where_clause
                else collection.query(
                    query_embeddings=[query_vector],
                    n_results=5,
                )
            )
            retrieved = "\n\n".join(results["documents"][0]) if results["documents"] else ""
        else:
            retrieved = ""
    except Exception as e:
        st.error(f"Vector DB query failed: {e}")
        retrieved = ""

    prompt = f"""
Use ONLY the following document context to answer. If answer not in context, reply exactly: "Not found in document."

CONTEXT:
{retrieved}

QUESTION:
{user_input}
"""

    answer = generate_with_gemini(prompt)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # re-render by forcing a short rerun to show new messages and scroll
    st.experimental_rerun()


# ================= RIGHT PANEL (Studio) ===================
with col_studio:
    st.header("🎛️ Studio Tools")

    if st.button("🧠 Summary"):
        if not st.session_state.active_doc:
            st.info("Select an active document first.")
        else:
            txt = generate_with_gemini(
                f"Summarize key points from the document named '{st.session_state.active_doc}' in 6 concise bullet points."
            )
            st.markdown(f"<div class='studio-card'>{txt}</div>", unsafe_allow_html=True)

    if st.button("🗺️ Mind Map"):
        if not st.session_state.active_doc:
            st.info("Select an active document first.")
        else:
            txt = generate_with_gemini(
                f"Create a simple text-based mind map (bullet hierarchy) for the document '{st.session_state.active_doc}'."
            )
            st.markdown(f"<div class='studio-card'>{txt}</div>", unsafe_allow_html=True)

    if st.button("📊 Report"):
        if not st.session_state.active_doc:
            st.info("Select an active document first.")
        else:
            txt = generate_with_gemini(
                f"Write a short structured report (Introduction, Key Points, Conclusion) for the document '{st.session_state.active_doc}'."
            )
            st.markdown(f"<div class='studio-card'>{txt}</div>", unsafe_allow_html=True)

    if st.button("❓ Quiz"):
        if not st.session_state.active_doc:
            st.info("Select an active document first.")
        else:
            txt = generate_with_gemini(
                f"Generate 5 multiple-choice questions (with answers) from the document '{st.session_state.active_doc}'."
            )
            st.markdown(f"<div class='studio-card'>{txt}</div>", unsafe_allow_html=True)
