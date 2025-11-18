# src/notebook_ui.py
import streamlit as st
import os
import tempfile
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import docx
import google.generativeai as genai
from google.generativeai import embed_content
from dotenv import load_dotenv

# ========== LOAD ENV ==========
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ========== CONFIG ==========
st.set_page_config(page_title="RAG Chatbot App", layout="wide")
DATA_DIR = Path(tempfile.gettempdir()) / "rag_uploads"
DATA_DIR.mkdir(exist_ok=True)

# ========== STYLING ==========
st.markdown(
    """
<style>
.block-container { padding-top: 1rem; }
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
.chat-message { margin: 6px 0; padding: 10px 12px; border-radius: 10px; max-width: 85%; word-wrap: break-word; white-space: pre-wrap; line-height: 1.4; }
.user-msg { background: #d9e8ff; align-self: flex-end; text-align: right; }
.bot-msg { background: #f2f4f7; align-self: flex-start; }
.studio-card { background: #fff; border-radius: 12px; padding: 12px; margin-bottom: 10px; box-shadow: 0 0 6px rgba(0,0,0,0.04); white-space: pre-wrap; }
.centered-input-wrapper { max-width: 900px; margin: 8px auto 36px auto; display: flex; gap: 8px; align-items: center; }
.source-list { font-size: 0.95em; margin-top:8px; color:#333;}
.source-item { padding:8px; border-radius:8px; background:#fafafa; margin-bottom:6px; box-shadow:0 0 3px rgba(0,0,0,0.03); }
</style>
""",
    unsafe_allow_html=True,
)

# ========== HELPERS: embedding + LLM ==========
def get_embedding(text: str) -> List[float]:
    resp = embed_content(model="models/text-embedding-004", content=text)
    emb = resp.get("embedding")
    if emb is None:
        raise RuntimeError("Embedding API returned no embedding.")
    return emb

def safe_gemini_response(response) -> str:
    try:
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        if hasattr(response, "candidates") and response.candidates:
            cand = response.candidates[0]
            if getattr(cand, "content", None) and getattr(cand.content, "parts", None):
                part = cand.content.parts[0]
                return getattr(part, "text", "").strip() or "⚠️ Empty part text."
        return "⚠️ The model returned no usable text (possibly blocked)."
    except Exception as e:
        return f"⚠️ Error extracting model response: {e}"

def generate_with_gemini(prompt: str, temperature: float = 0.2) -> str:
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(prompt, generation_config={"temperature": temperature, "max_output_tokens": 900})
        return safe_gemini_response(response)
    except Exception as e:
        return f"⚠️ Gemini API error: {e}"

# -------- DOCUMENT READERS / CHUNKER --------
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

def chunk_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
    return splitter.split_text(text)

# -------- FAISS / STORE MANAGEMENT --------
def ensure_faiss_state():
    if "faiss_index" not in st.session_state:
        st.session_state.faiss_index = None
    if "faiss_dim" not in st.session_state:
        st.session_state.faiss_dim = None
    if "docstore" not in st.session_state:
        st.session_state.docstore = []  # list of {"id","chunk","source","chunk_index"}

def add_embeddings_to_faiss(embeddings: List[List[float]], chunks: List[str], source: str):
    ensure_faiss_state()
    if not embeddings:
        return
    arr = np.array(embeddings, dtype="float32")
    dim = arr.shape[1]
    if st.session_state.faiss_index is None:
        st.session_state.faiss_index = faiss.IndexFlatL2(dim)
        st.session_state.faiss_dim = dim
    elif st.session_state.faiss_dim != dim:
        raise RuntimeError(f"Embedding dim mismatch: {st.session_state.faiss_dim} vs {dim}")
    st.session_state.faiss_index.add(arr)
    start_idx = len(st.session_state.docstore)
    for i, chunk in enumerate(chunks):
        st.session_state.docstore.append({"id": f"{source}_{start_idx + i}", "chunk": chunk, "source": source, "chunk_index": start_idx + i})

def query_faiss(query_embedding: List[float], top_k: int = 8) -> List[Dict]:
    ensure_faiss_state()
    if st.session_state.faiss_index is None:
        return []
    q = np.array(query_embedding, dtype="float32").reshape(1, -1)
    D, I = st.session_state.faiss_index.search(q, top_k)
    indices = I[0].tolist()
    results = []
    for idx in indices:
        if idx < 0 or idx >= len(st.session_state.docstore):
            continue
        results.append(st.session_state.docstore[idx])
    return results

# -------- INIT SESSION ----------
ensure_faiss_state()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []
if "active_doc" not in st.session_state:
    st.session_state.active_doc = None

st.title("💬 RAG Chatbot App")
st.caption("Built by *Ravi Rajak* — answers with source snippets")

col_sources, col_chat, col_studio = st.columns([1.5, 2.5, 1.5])

# ---------------- LEFT: upload & select ---------------
with col_sources:
    st.header("📚 Upload & Sources")
    uploaded_files = st.file_uploader("Upload up to 20 documents (pdf/docx/txt)", type=["pdf","docx","txt"], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files[:20]:
            save_path = DATA_DIR / file.name
            with open(save_path, "wb") as f:
                f.write(file.read())
            if file.name not in st.session_state.uploaded_docs:
                st.session_state.uploaded_docs.append(file.name)
            ext = file.name.split(".")[-1].lower()
            text = read_pdf(save_path) if ext == "pdf" else read_docx(save_path) if ext == "docx" else read_txt(save_path)
            chunks = chunk_text(text)
            embeddings = []
            for chunk in chunks:
                try:
                    embeddings.append(get_embedding(chunk))
                except Exception as e:
                    st.error(f"Embedding failed for a chunk: {e}")
                    if st.session_state.faiss_dim:
                        embeddings.append([0.0]*st.session_state.faiss_dim)
                    else:
                        embeddings.append([0.0])
            add_embeddings_to_faiss(embeddings, chunks, file.name)
        st.success("✅ Files uploaded & indexed!")
    if st.session_state.uploaded_docs:
        st.session_state.active_doc = st.selectbox("📄 Currently Working On", st.session_state.uploaded_docs, index=0 if st.session_state.active_doc is None else st.session_state.uploaded_docs.index(st.session_state.active_doc))
    else:
        st.info("Upload documents to get started.")

# ---------------- CENTER: chat display ----------------
with col_chat:
    st.header("💬 Chat")
    chat_box = st.empty()
    html = "<div class='chat-wrapper' id='chat-box'>"
    for msg in st.session_state.messages:
        cls = "user-msg" if msg["role"]=="user" else "bot-msg"
        sender = "🧑‍💻 You" if msg["role"]=="user" else "🤖 Bot"
        content = str(msg["content"]).replace("\n","<br>")
        html += f"<div class='chat-message {cls}'><b>{sender}:</b><br>{content}</div>"
    html += "</div>"
    chat_box.markdown(html, unsafe_allow_html=True)
    st.markdown("""
    <script>
    (function(){ try { const parent = window.parent.document; const box = parent.querySelector('#chat-box'); if (box) box.scrollTop = box.scrollHeight; } catch(e){} })()
    </script>
    """, unsafe_allow_html=True)
    st.markdown("---")

# ---------------- chat input (root level, visually centered) ----------------
st.markdown("<div class='centered-input-wrapper'>", unsafe_allow_html=True)
user_input = st.chat_input("Ask a question...")
col_for_clear = st.columns([1,1,1,1,1])[4]
with col_for_clear:
    if st.button("🧹 Clear"):
        st.session_state.messages = []
        st.experimental_rerun()
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- process user query and produce answer + sources (detailed) ----------------
if user_input:
    st.session_state.messages.append({"role":"user","content":user_input})
    try:
        q_emb = get_embedding(user_input)
    except Exception as e:
        st.error(f"Failed to create query embedding: {e}")
        q_emb = None

    retrieved = []
    if q_emb is not None:
        # get nearest neighbors
        hits = query_faiss(q_emb, top_k=10)
        # apply active_doc filter client-side and keep order
        if st.session_state.active_doc:
            hits = [h for h in hits if h.get("source")==st.session_state.active_doc]
            if not hits:
                # fallback to unfiltered if none after filter
                hits = query_faiss(q_emb, top_k=5)
        retrieved = hits[:6]  # keep top 6 chunks

    # Build RAG prompt using retrieved chunks
    context_pieces = []
    for item in retrieved:
        t = item.get("chunk","")
        context_pieces.append(t)
    context_for_prompt = "\n\n".join(context_pieces) if context_pieces else ""

    prompt = f"""You are an assistant that MUST answer using only the given CONTEXT. If the answer cannot be found in the context, reply exactly: "Not found in document."

CONTEXT:
{context_for_prompt}

QUESTION:
{user_input}
"""
    answer_text = generate_with_gemini(prompt)

    # Build a structured source attribution (detailed with snippets)
    # aggregate unique sources in order of appearance
    sources_ordered = []
    source_snips = {}
    for item in retrieved:
        src = item.get("source")
        idx = item.get("chunk_index")
        chunk_text = item.get("chunk","")
        if src not in source_snips:
            # snippet: first 150 chars (trim)
            snippet = chunk_text.strip().replace("\n"," ")
            snippet = (snippet[:150] + "...") if len(snippet) > 150 else snippet
            source_snips[src] = {"snippet": snippet, "chunk_index": idx}
            sources_ordered.append(src)

    # Compose assistant message content: answer + Sources block with snippet previews
    sources_block_lines = []
    for s in sources_ordered:
        ci = source_snips[s]["chunk_index"]
        snip = source_snips[s]["snippet"]
        sources_block_lines.append(f"- **{s}** (chunk #{ci}): \"{snip}\"")

    sources_block = "\n".join(sources_block_lines) if sources_block_lines else "No supporting sources found."

    assistant_full = f"{answer_text}\n\n📄 **Sources & snippets used:**\n{sources_block}"

    st.session_state.messages.append({"role":"assistant","content":assistant_full})
    st.experimental_rerun()

# ---------------- RIGHT: Studio Tools ----------------
with col_studio:
    st.header("🎛️ Studio Tools")
    if st.button("🧠 Summary"):
        if not st.session_state.active_doc:
            st.info("Select an active document first.")
        else:
            prompt = f"Summarize key points from the document named '{st.session_state.active_doc}' in 6 concise bullet points."
            txt = generate_with_gemini(prompt)
            st.markdown(f"<div class='studio-card'>{txt}</div>", unsafe_allow_html=True)

    if st.button("🗺️ Mind Map"):
        if not st.session_state.active_doc:
            st.info("Select an active document first.")
        else:
            prompt = f"Create a mind map (bullet hierarchy) for the document '{st.session_state.active_doc}'."
            txt = generate_with_gemini(prompt)
            st.markdown(f"<div class='studio-card'>{txt}</div>", unsafe_allow_html=True)

    if st.button("📊 Report"):
        if not st.session_state.active_doc:
            st.info("Select an active document first.")
        else:
            prompt = f"Write a short structured report (Introduction, Key Points, Conclusion) for the document '{st.session_state.active_doc}'."
            txt = generate_with_gemini(prompt)
            st.markdown(f"<div class='studio-card'>{txt}</div>", unsafe_allow_html=True)

    if st.button("❓ Quiz"):
        if not st.session_state.active_doc:
            st.info("Select an active document first.")
        else:
            prompt = f"Generate 5 multiple-choice questions (with answers) from the document '{st.session_state.active_doc}'."
            txt = generate_with_gemini(prompt)
            st.markdown(f"<div class='studio-card'>{txt}</div>", unsafe_allow_html=True)
