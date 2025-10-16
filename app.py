import os
import io
import pickle
import faiss
import numpy as np
import streamlit as st
from PIL import Image
from pypdf import PdfReader
from pdf2image import convert_from_path
import easyocr
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import torch

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="RAG Q&A (PDF + Image)", page_icon="📚")

EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_FILE = "vector_store.pkl"

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# UTILS
# -----------------------------
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    # If PDF is scanned, fall back to OCR
    if not text.strip():
        images = convert_from_path(file)
        reader_ocr = easyocr.Reader(["en"], verbose=False)
        for img in images:
            results = reader_ocr.readtext(np.array(img), detail=0)
            text += " ".join(results) + " "
    return text.strip() or "[No readable text found]"

def extract_text_from_image(file):
    image = Image.open(file).convert("RGB")
    reader = easyocr.Reader(["en"], verbose=False)
    results = reader.readtext(np.array(image), detail=0)
    return " ".join(results).strip() or "[No readable text found]"

def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def build_index(texts):
    model = SentenceTransformer(EMBED_MODEL)
    chunks = []
    for doc in texts:
        chunks.extend(split_text(doc))
    embeddings = model.encode(chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    store = {"index": index, "chunks": chunks, "model": model}
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(store, f)
    return store

def retrieve(query, store, top_k=3):
    q_emb = store["model"].encode([query], convert_to_numpy=True)
    D, I = store["index"].search(q_emb, top_k)
    return [store["chunks"][i] for i in I[0]]

# -----------------------------
# OPTIONAL: CLIP for image understanding
# -----------------------------
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_image_labels(image: Image.Image, candidate_labels=["cat","dog","flower","document","person","car"]):
    inputs = clip_processor(text=candidate_labels, images=image, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    image_features = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_features = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
    similarity = (image_features @ text_features.T).squeeze(0)
    top_idx = similarity.argmax().item()
    return candidate_labels[top_idx]

# -----------------------------
# STREAMLIT APP
# -----------------------------
st.title("📚 RAG Q&A (PDF + Image)")
st.info("Upload PDF or image files. Ask any question and get answers from the uploaded content!")

uploaded_files = st.file_uploader("Upload PDFs or Images", type=["pdf","png","jpg","jpeg"], accept_multiple_files=True)

if uploaded_files:
    all_texts = []
    for file in uploaded_files:
        if file.type == "application/pdf":
            text = extract_text_from_pdf(file)
            st.success(f"✅ Extracted text from PDF: {file.name}")
        else:
            text = extract_text_from_image(file)
            st.write(f"✅ Extracted text preview ({file.name}): {text[:500]}...")
            # Optional: get image label via CLIP
            label = get_image_labels(Image.open(file))
            st.write(f"🔹 Detected image content: {label}")
        all_texts.append(text)

    if st.button("🔍 Build Knowledge Base"):
        with st.spinner("Creating embeddings..."):
            store = build_index(all_texts)
            st.success("✅ Knowledge base created!")

    query = st.text_input("Ask a question about the uploaded files:")
    if query and os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "rb") as f:
            store = pickle.load(f)
        contexts = retrieve(query, store)
        st.subheader("🔍 Retrieved Contexts")
        for i, ctx in enumerate(contexts, 1):
            st.write(f"**Context {i}:** {ctx[:400]}...")

        # Simple answer synthesis using the retrieved context
        answer = " ".join(contexts)  # For local, you can plug in a small LLM here if desired
        st.subheader("🧠 Answer (from context)")
        st.write(answer)
