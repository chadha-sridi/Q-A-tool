# index_pdf.py
import pickle
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

#config
PDF_PATH = "pnds_ahai_de_ladulte_version_2024_nobib.pdf"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
SBERT_MODEL = "all-mpnet-base-v2"


def pdf_to_text(file_path: str):
    reader = PdfReader(file_path)
    pages = []
    for p in reader.pages:
        try:
            text = p.extract_text() or ""
        except Exception:
            text = ""
        pages.append(text)
    return pages

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunks.append(" ".join(tokens[i:i+chunk_size]))
        i += chunk_size - overlap
    return chunks
# Load SBERT and index

sbert = SentenceTransformer(SBERT_MODEL)

def embed_texts(texts):
    embs = sbert.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embs.astype(np.float32)

# Process PDF
pages = pdf_to_text(PDF_PATH)
chunk_texts = []
for page_idx, page in enumerate(pages):
    if not page.strip():
        continue
    chunks = chunk_text(page)
    for chunk_idx, c in enumerate(chunks):
        chunk_texts.append({"text": c, "page": page_idx + 1, "chunk_id": f"{page_idx+1}_{chunk_idx}"})

texts = [c["text"] for c in chunk_texts]
embs = embed_texts(texts)

# Normalize for cosine similarity
embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
# FAISS index

EMBED_DIM = embs.shape[1]
index = faiss.IndexFlatIP(EMBED_DIM)
index.add(embs)

# Save index and metadata
faiss.write_index(index, "faiss_index.index")
with open("metas.pkl", "wb") as f:
    pickle.dump(chunk_texts, f)
breakpoint()
print(f"Indexed {len(chunk_texts)} chunks from {len(pages)} pages.")
