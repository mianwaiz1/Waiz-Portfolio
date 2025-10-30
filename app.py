import os
import numpy as np
from flask import Flask, render_template, request, jsonify, session, abort
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from flask_session import Session
from werkzeug.middleware.proxy_fix import ProxyFix

# ---------- SETUP ----------
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file.")
os.environ["GOOGLE_API_KEY"] = google_api_key

# Flask App
app = Flask(__name__)

# Load secret key securely
app.secret_key = os.getenv("SECRET_KEY", os.urandom(24))

# Secure session config
app.config.update(
    SESSION_TYPE="filesystem",
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=True,   # ensure HTTPS in production
    SESSION_COOKIE_SAMESITE="Lax",
    MAX_CONTENT_LENGTH=2 * 1024 * 1024,  # 2 MB request limit
)

Session(app)
app.wsgi_app = ProxyFix(app.wsgi_app)  # handle proxies correctly

# ---------- LLM + EMBEDDING ----------
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---------- PDF LOADING ----------
PDF_PATH = "data/Resume.pdf"

def load_vectorstore():
    loader = PyMuPDFLoader(PDF_PATH)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embedding=embedder)
    return vectorstore, chunks

print("📄 Loading PDF and building vectorstore...")
vectorstore, chunks = load_vectorstore()
print("✅ PDF loaded successfully!")

# ---------- HELPER FUNCTIONS ----------
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

    history_text = ""
    if chat_history:
        history_text = "\n".join(
            [f"User: {msg['user']}\nBot: {msg['bot']}" for msg in chat_history[-3:]]
        )

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

# ---------- SECURITY HEADERS ----------
@app.after_request
def add_security_headers(response):
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';"
    )
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=()"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

# ---------- ROUTES ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")

@app.route("/ask", methods=["POST"])
def ask():
    if not request.is_json:
        abort(400, "Invalid request format. JSON required.")
    data = request.get_json()
    query = str(data.get("question", "")).strip()
    if not query:
        return jsonify({"error": "Empty question"}), 400

    if "chat_history" not in session:
        session["chat_history"] = []

    try:
        answer, score = get_answer(query, session["chat_history"])
        session["chat_history"].append({"user": query, "bot": answer})
        session.modified = True
        return jsonify({"answer": answer, "score": score})
    except Exception as e:
        return jsonify({"error": "An internal error occurred."}), 500

@app.route("/clear", methods=["POST"])
def clear_history():
    session.pop("chat_history", None)
    return jsonify({"status": "cleared"})

# ---------- MAIN ----------
if __name__ == "__main__":
    # Never expose debug=True publicly
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=5000, debug=debug_mode)
