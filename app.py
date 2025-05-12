from flask import Flask, request, render_template, session
from flask_session import Session
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

import fitz  # PyMuPDF
import tempfile

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Güvenlik için bir anahtar
app.config["SESSION_TYPE"] = "filesystem"  # Session tipi
Session(app)

@app.route("/")
def index():
    # Session'dan sohbet geçmişini al
    chat_history = session.get("chat_history", [])
    return render_template("index.html", chat_history=chat_history)

@app.route("/ask", methods=["POST"])
def ask():
    # Global chat geçmişini oturumda sakla
    file = request.files["file"]
    question = request.form["question"]
    chat_history = session.get("chat_history", [])

    # PDF içeriğini oku
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        file.save(tmp.name)
        doc = fitz.open(tmp.name)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()

    # Belgeyi indexleyelim
    document = Document(text=text)
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = Ollama(model="llama3")
    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents([document])
    index = VectorStoreIndex(nodes, embed_model=embed_model)
    query_engine = index.as_query_engine(llm=llm)

    # Prompt'u dilde tutalım, ancak text'i direkt vermiyoruz
    response = query_engine.query(question)

    # Sohbet geçmişini güncelle
    chat_history.append(("user", question))
    chat_history.append(("bot", response.response))

    # Yeni sohbet geçmişini session'a kaydet
    session["chat_history"] = chat_history

    return render_template("index.html", chat_history=chat_history)

if __name__ == "__main__":
    app.run(debug=True)
