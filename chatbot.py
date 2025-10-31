import os
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# ------------------ SETUP ------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not found in .env")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

st.set_page_config(page_title="Waiz Khalil — Personal Chatbot", layout="wide")

# ------------------ CSS (replicates your HTML exactly) ------------------
st.markdown(
    """
    <style>
    /* Reset streamlit chrome */
    header, footer, .stApp, .css-1d391kg { padding: 0; margin: 0; }
    .block-container { padding: 0 0 0 0; margin: 0; }

    :root {
      --bg: #0f1724;
      --muted: #9aa4b2;
      --accent: #6ee7b7;
      --user: #60a5fa;
      --radius: 14px;
    }

    html, body {
      margin: 0;
      height: 100%;
      font-family: "Inter", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
      background: linear-gradient(180deg, #071427 0%, #081428 100%);
      color: #e6eef6;
    }

    /* hide streamlit header bar etc */
    .css-1lsmgbg { display:none !important; }
    .css-18e3th9 { padding: 0 !important; } /* main*/
    .stButton>button { outline: none; }

    /* App layout */
    .page {
      display:flex;
      flex-direction:column;
      min-height:5vh;
      justify-content:space-between;
    }

    header.custom {
      padding: 16px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid rgba(255,255,255,0.05);
      background: rgba(10,18,32,0.6);
      backdrop-filter: blur(8px);
    }

    .brand { display:flex; align-items:center; gap:12px; }
    .avatar {
      width:48px; height:48px; border-radius:12px;
      background: linear-gradient(135deg, var(--accent), #60a5fa);
      display:grid; place-items:center; color:#072033; font-weight:700; font-size:18px;
    }

    main.custom {
      flex:1; display:flex; flex-direction:column; align-items:center;
      justify-content:flex-start; padding:20px; overflow:auto;
    }

    #chatlog {
        width: 100%;
        max-width: 800px;
        height: 60vh;                 /* fixed visible height */
        overflow-y: auto;
        padding: 20px;
        border-radius: 12px;
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        margin: 16px auto;
        display: flex;
        flex-direction: column;
        gap: 14px;
        scroll-behavior: smooth;
    }


    .msg { display:inline-block; max-width:80%; padding:10px 14px; border-radius:12px;
      line-height:1.5; word-wrap:break-word; font-size:14px; animation:fadeIn .2s ease-in;
    }
    .user { align-self:flex-end; background:rgba(96,165,250,0.15);
      border:1px solid rgba(96,165,250,0.3); color:var(--user);
    }
    .bot { align-self:flex-start; background:rgba(110,231,183,0.1);
      border:1px solid rgba(110,231,183,0.2); color:var(--accent);
    }
    @keyframes fadeIn { from {opacity:0; transform:translateY(4px)} to {opacity:1; transform:translateY(0)} }

    .input-area {
        display: flex;
        gap: 8px;
        width: 100%;
        max-width: 800px;
        position: fixed;          /* FIXED POSITION */
        bottom: 20px;             /* spacing from bottom */
        left: 50%;
        transform: translateX(-50%);
        padding: 12px 16px;
        background: rgba(10,18,32,0.85);
        backdrop-filter: blur(8px);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.05);
        z-index: 999;
    }

    .input-area input {
      flex:1; padding:12px 14px; border-radius:10px;
      border:1px solid rgba(255,255,255,0.08); background: rgba(255,255,255,0.05);
      color:#fff; font-size:14px; outline:none;
    }
    .send-btn {
      background:var(--accent); border:none; color:#072033;
      padding:10px 18px; border-radius:8px; cursor:pointer; font-weight:600;
    }
    .send-btn:hover { background:#34d399; }
    footer.custom {
      text-align:center; font-size:13px; color:var(--muted);
      padding:12px; border-top:1px solid rgba(255,255,255,0.05);
      background: rgba(10,18,32,0.6);
    }
    @media (max-width:600px) { #chatlog {max-width:100%} .msg {max-width:95%} }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------ MODELS (same logic) ------------------
@st.cache_resource
def init_models():
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return llm, embedder

llm, embedder = init_models()

# ------------------ LOAD PDF & VECTORSTORE (same logic) ------------------
PDF_PATH = "data/Resume.pdf"
@st.cache_resource
def load_vectorstore():
    loader = PyMuPDFLoader(PDF_PATH)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embedding=embedder)
    return vectorstore, chunks

try:
    vectorstore, chunks = load_vectorstore()
except Exception as e:
    st.error(f"Failed loading PDF/vectorstore: {e}")
    st.stop()

# ------------------ helper functions (identical semantics) ------------------
def semantic_score(answer, references):
    try:
        ans_emb = embedder.embed_query(answer)
        ref_embs = [embedder.embed_query(r) for r in references]
        sims = [
            np.dot(ans_emb, ref_emb) / (np.linalg.norm(ans_emb) * np.linalg.norm(ref_emb))
            for ref_emb in ref_embs
        ]
        return round(max(sims) * 100, 2)
    except Exception:
        return 0.0

def get_answer(query, chat_history=None, top_k=4):
    docs = vectorstore.similarity_search(query, k=top_k)
    refs = [d.page_content for d in docs]
    context = "\n\n".join(refs)

    # build history string only when both user+bot pairs exist in history
    history_text = ""
    if chat_history:
        # chat_history expected to be list of dicts like {"user": "...", "bot": "..."}
        for pair in chat_history[-3:]:
            if isinstance(pair, dict) and "user" in pair and "bot" in pair:
                history_text += f"User: {pair['user']}\nBot: {pair['bot']}\n"

    prompt = f"""
You are made by Waiz. He is your owner. You have his information. Waiz is a human and your owner who made you talk with others about him.
You are WaizBot — a warm, professional, and friendly AI assistant. 
You help users by answering questions clearly and conversationally, based on the provided data.
Always stay grounded in the given context, but phrase answers in a natural and human way.
Use short paragraphs or bullet points for clarity when helpful.

Past Conversation (for continuity):
{history_text}

Relevant Document Excerpts:
{context}

Now, answer this question thoughtfully and conversationally:
{query}
"""
    answer = llm.invoke(prompt).content
    score = semantic_score(answer, refs)
    return answer, score

# ------------------ session_state ------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # will store dicts {"user":..., "bot":...}

# ------------------ FRONTEND: render exact HTML-like layout ------------------
# Note: We use streamlit's markdown to render the exact layout, and st.form for single input.
st.markdown(
    """
    <div class="page">
      <header class="custom">
        <div class="brand">
          <div class="avatar">WK</div>
          <div>
            <h1 style="margin:0;font-size:18px;">Waiz Khalil</h1>
            <div style="font-size:13px;color:var(--muted);">Personal AI Chatbot</div>
          </div>
        </div>
        <nav><a href="https://waizkhalil.pythonanywhere.com/">← Back to Portfolio</a></nav>
      </header>
    </div>
    """,
    unsafe_allow_html=True,
)

# Render chat messages inside the chatlog div (we append HTML blocks)
chat_html_parts = []
for pair in st.session_state.chat_history:
    # pair expected {"user": "...", "bot": "..."}
    if "user" in pair:
        chat_html_parts.append(f'<div class="msg user"><b>You:</b> {st.session_state.get("_escape", lambda x: x)(pair["user"])}</div>')
    if "bot" in pair:
        chat_html_parts.append(f'<div class="msg bot"><b>WaizBot:</b> {st.session_state.get("_escape", lambda x: x)(pair["bot"])}</div>')

# Combine and render
chat_html = "\n".join(chat_html_parts) if chat_html_parts else ""
st.markdown(f'<div id="chatlog" class="chatbox-contents">{chat_html}</div>', unsafe_allow_html=True)

# Input area (single)
with st.form("waiz_form", clear_on_submit=True):
    user_input = st.text_input("", placeholder="Ask something about Waiz...", label_visibility="collapsed", key="input_field")
    send_btn = st.form_submit_button("Send")
    clear_btn = st.form_submit_button("Clear")
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------ Actions ------------------
if clear_btn:
    st.session_state.chat_history = []
    # rerun to update UI immediately
    st.rerun()

if send_btn and user_input and user_input.strip():
    query = user_input.strip()
    # append user message as a pair with bot empty for now,
    # we store pairs only when bot answers, to match original history format.
    # We'll temporarily show the user message first by adding a pair with user only,
    # then replace/extend with bot answer after the model returns.
    st.session_state.chat_history.append({"user": query})
    # re-render UI immediately (optional)
    st.rerun()

# When a user message exists at the end without a bot reply, call the LLM and append the bot reply.
# This lets the UI show only one input and the messages flow properly.
if st.session_state.chat_history:
    last = st.session_state.chat_history[-1]
    # If last entry only has "user" and no "bot", produce reply now.
    if isinstance(last, dict) and "user" in last and "bot" not in last:
        user_q = last["user"]
        with st.spinner("Thinking..."):
            try:
                answer, score = get_answer(user_q, st.session_state.chat_history[:-1])
            except Exception as e:
                answer = "Sorry — internal error when generating response."
            # update the last pair to include bot
            st.session_state.chat_history[-1] = {"user": user_q, "bot": answer}
        # re-run to show the answer in the chatlog
        st.rerun()

# Footer
st.markdown('<footer class="custom">Made with ❤️ by Waiz</footer></div>', unsafe_allow_html=True)

