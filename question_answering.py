# qa_flan_t5.py
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import faiss
from sentence_transformers import SentenceTransformer
from transformers import logging
logging.set_verbosity_error()

TOP_K = 4
MAX_INPUT_CHARS = 1500
FLAN_MODEL = "google/flan-t5-base"
SBERT_MODEL = "all-mpnet-base-v2"
FAISS_FILE = "faiss_index.index"
METAS_FILE = "metas.pkl"

if not (os.path.exists(FAISS_FILE) and os.path.exists(METAS_FILE)):
        print("FAISS index or metadata not found. Running index_pdf.py...")
        subprocess.run(["python", "index_pdf.py"], check=True)
# Load FAISS & metadata
index = faiss.read_index("faiss_index.index")
with open("metas.pkl", "rb") as f:
    metas = pickle.load(f)

# Load models
sbert = SentenceTransformer(SBERT_MODEL)
tokenizer = AutoTokenizer.from_pretrained(FLAN_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(FLAN_MODEL)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)

def normalize(vecs):
    return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10)

def embed_texts(texts):
    embs = sbert.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embs.astype(np.float32)

def summarize_chunk(text):
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def retrieve(query, top_k=TOP_K):
    q_emb = embed_texts([query])
    q_emb = normalize(q_emb)
    D, I = index.search(q_emb, top_k)
    results = [metas[i] for i in I[0]]
    return results

def build_prompt(question, retrieved):
    summaries = [summarize_chunk(r["text"]) for r in retrieved]
    context = "\n\n".join([f"[page:{r['page']}] {s}" for r, s in zip(retrieved, summaries)])
    prompt = (
        f"Tu es un assistant expert en santé. Utilise uniquement les extraits fournis.\n"
        f"QUESTION: {question}\n\n"
        f"CONTEXT:\n{context}\n"
        "Réponds uniquement en français, synthétique, et cite les sources. "
        "Si l'information n'est pas dans les extraits, réponds: 'Je n'ai trouvé aucune information pertinente.'"
    )
    return prompt[:MAX_INPUT_CHARS]

def answer_question(question):
    retrieved = retrieve(question)
    if not retrieved:
        return "Aucun contenu indexé."
    prompt = build_prompt(question, retrieved)
    out = summarizer(prompt, max_length=256, min_length=50, do_sample=False)
    return out[0]['summary_text']

print("=== Document Q&A ===\n")
while True:
    question = input("Pose ta question (ou 'exit' pour quitter): ").strip()
    if question.lower() == "exit":
        break
    answer = answer_question(question)
    print("\n=== Réponse ===")
    print(answer)
    print("\n--------------------\n")
