import os
import streamlit as st
from pypdf import PdfReader
from PIL import Image
import numpy as np
import easyocr
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import openai

# ========== CONFIG ==========
openai.api_key = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_FILE = "vector_store.pkl"
UPLOAD_DIR = "uploaded_docs"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ========== HELPERS ==========
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def extract_text_from_image(file):
    reader = easyocr.Reader(["en"], gpu=False)
    image = Image.open(file)
    result = reader.readtext(np.array(image))
    return " ".join([text for (_, text, _) in result])

def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

@st.cache_resource
def load_or_create_index(all_texts):
    model = SentenceTransformer(EMBED_MODEL)
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "rb") as f:
            store = pickle.load(f)
        return store, model

    chunks = []
    for text in all_texts:
        chunks.extend(split_text(text))
    embeddings = model.encode(chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    store = {"index": index, "chunks": chunks}
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(store, f)
    return store, model

def retrieve(query, store, model, top_k=3):
    q_emb = model.encode([query], convert_to_numpy=True)
    D, I = store["index"].search(q_emb, top_k)
    return [store["chunks"][i] for i in I[0]]

def synthesize_answer(query, contexts):
    context_text = "\n\n".join(contexts)
    prompt = f"Using the following context, answer concisely:\n{context_text}\n\nQuestion: {query}\nAnswer:"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# ========== STREAMLIT UI ==========
st.title("📚 Knowledge-base Search Engine (PDF + Image Upload)")

st.markdown("Upload **PDFs or images**. The app will extract text and build a searchable knowledge base.")

uploaded_files = st.file_uploader(
    "Upload documents (PDF or image):",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if uploaded_files:
    all_texts = []
    for file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())

        if file.name.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        else:
            text = extract_text_from_image(file_path)
        all_texts.append(text)

    st.success(f"{len(uploaded_files)} files processed successfully!")
    store, model = load_or_create_index(all_texts)

    query = st.text_input("Ask a question about your documents:")
    if st.button("Search"):
        if query:
            with st.spinner("Searching relevant content..."):
                contexts = retrieve(query, store, model)
                st.subheader("🔍 Retrieved Contexts")
                for i, ctx in enumerate(contexts, 1):
                    st.markdown(f"**Context {i}:** {ctx[:400]}...")

            with st.spinner("Synthesizing answer..."):
                answer = synthesize_answer(query, contexts)
                st.subheader("🧠 Synthesized Answer")
                st.write(answer)
        else:
            st.warning("Please enter a question to search.")
else:
    st.info("Please upload at least one file to start.")
