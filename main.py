import io, os, numpy as np, pandas as pd, requests, faiss
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ---------- Loaders ----------
def load_txt(file_bytes, name):
    return [{"source": name, "text": file_bytes.decode('utf-8', errors='ignore')}]

def load_pdf(file_bytes, name):
    r = PdfReader(io.BytesIO(file_bytes))
    text = "".join([(p.extract_text() or "") for p in r.pages])
    return [{"source": name, "text": text}]

def load_docx(file_bytes, name):
    doc = Document(io.BytesIO(file_bytes))
    text = " ".join(p.text for p in doc.paragraphs)
    return [{"source": name, "text": text}]

def load_csv(file_bytes, name):
    df = pd.read_csv(io.BytesIO(file_bytes))
    text = " ".join(df.astype(str).values.flatten())
    return [{"source": name, "text": text}]

def load_documents_from_paths(paths):
    docs = []
    for p in paths:
        with open(p, 'rb') as f:
            data = f.read()
        pl = p.lower()
        if pl.endswith('.pdf'): docs += load_pdf(data, os.path.basename(p))
        elif pl.endswith('.docx'): docs += load_docx(data, os.path.basename(p))
        elif pl.endswith('.csv'): docs += load_csv(data, os.path.basename(p))
        elif pl.endswith('.txt'): docs += load_txt(data, os.path.basename(p))
    return docs

def fetch_web_text(url):
    try:
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        for s in soup(["script", "style", "noscript"]): s.decompose()
        return [{"source": url, "text": soup.get_text(separator=' ')}]
    except Exception as e:
        print("Failed:", url, e); return []

def load_documents_from_urls(urls):
    docs = []
    for u in urls or []:
        u = u.strip()
        if u: docs += fetch_web_text(u)
    return docs

# ---------- Chunking ----------
def chunk_documents(docs, chunk_size=1000, chunk_overlap=100):
    chunks = []
    for doc in docs:
        t = doc["text"] or ""
        start, idx = 0, 0
        while start < len(t):
            end = start + chunk_size
            chunks.append({"source": doc["source"], "chunk_id": f"{doc['source']}_chunk{idx}", "content": t[start:end]})
            idx += 1
            start = max(0, end - chunk_overlap)
            if start == end: break
    return chunks

# ---------- Embeddings & FAISS ----------
_embedder = None
def load_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # 384-d [Hugging Face Pipelines doc ref]
    return _embedder

def get_embeddings(chunks, embedder):
    emb = embedder.encode([c['content'] for c in chunks], show_progress_bar=True, convert_to_numpy=True)
    return emb.astype('float32')

def build_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])  # exact L2 search
    index.add(embeddings)
    return index

def retrieve(query, embedder, index, chunks, top_k=3):
    q = embedder.encode([query], convert_to_numpy=True).astype('float32')
    D, I = index.search(q, top_k)
    # Iterate through the indices of the first query result
    return [{"chunk": chunks[i], "score": float(D[0][r])} for r, i in enumerate(I[0]) if i != -1]

# ---------- LLM ----------
_llm = None
def load_llm():
    global _llm
    if _llm is None:
        _llm = pipeline(
            "text-generation",
            model="MehdiHosseiniMoghadam/AVA-Mistral-7B-V2",
            tokenizer="MehdiHosseiniMoghadam/AVA-Mistral-7B-V2",
            max_length=1024,
            do_sample=True,
            temperature=0.2,
            trust_remote_code=True,
            device_map="auto"  # use GPU if available
        )
    return _llm

def answer_with_llm(context_chunks, query, llm):
    ctx = "\n".join([f"[{c['chunk_id']}] {c['content']}" for c in context_chunks])
    prompt = (
        "Answer the following question using ONLY the provided context and cite the chunk ids used.\n"
        f"Question: {query}\nContext:\n{ctx}\nAnswer with citations:"
    )
    out = llm(prompt, max_new_tokens=512, num_return_sequences=1)
    return out[0]['generated_text'] if isinstance(out, list) and out else str(out)

# ---------- Run once ----------
# Optionally upload files via Colab UI, then set file_paths = ["/content/your.pdf", ...]
file_paths = []  # e.g., ["/content/sample.txt"]
urls = ["https://huggingface.co/docs/transformers/en/main_classes/pipelines"]  # change or set []

query = "How do I use the text-generation pipeline?"  # edit your question

# Pipeline
file_docs = load_documents_from_paths(file_paths)
web_docs = load_documents_from_urls(urls)
all_docs = file_docs + web_docs
assert all_docs, "No documents. Upload files or set URLs."

chunks = chunk_documents(all_docs, chunk_size=1000, chunk_overlap=100)
embedder = load_embedder()
emb = get_embeddings(chunks, embedder)
index = build_faiss_index(emb)
llm = load_llm()
retrieved = retrieve(query, embedder, index, chunks, top_k=3)
answer = answer_with_llm([r["chunk"] for r in retrieved], query, llm)

print("\n=== Answer ===\n", answer)
print("\n=== Sources ===")
for r in retrieved:
    print(f"[{r['chunk']['chunk_id']}] from {r['chunk']['source']} (score={r['score']:.4f})")
