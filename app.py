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

# ================= CONFIG =================
openai.api_key = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = "all-MiniLM-L6-v2"

# ================= HELPERS =================

def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def split_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

@st.cache_resource
def get_easyocr_reader():
    return easyocr.Reader(["en"], gpu=False)

@st.cache_resource
def get_sentence_model():
    return SentenceTransformer(EMBED_MODEL)

def extract_text_from_image(file, reader):
    image = Image.open(file)
    result = reader.readtext(np.array(image))
    return " ".join([text for (_, text, _) in result])

def create_index(all_texts, model):
    chunks = []
    for text in all_texts:
        chunks.extend(split_text(text))
    embeddings = model.encode(chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    store = {"index": index, "chunks": chunks}
    return store

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

# ================= STREAMLIT UI =================
st.title("📚 Knowledge-base Search Engine (PDF + Image Upload)")

st.markdown("Upload **PDFs or images**. The app will extract text and build a searchable knowledge base.")

uploaded_files = st.file_uploader(
    "Upload documents (PDF or image):",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

# Initialize session_state variables
if "store" not in st.session_state:
    st.session_state.store = None
if "all_texts" not in st.session_state:
    st.session_state.all_texts = []

model = get_sentence_model()
reader = get_easyocr_reader()

# Process uploaded files
if uploaded_files:
    new_texts = []
    for file in uploaded_files:
        if file.name.endswith(".pdf"):
            text = extract_text_from_pdf(file)
        else:
            text = extract_text_from_image(file, reader)
        if text.strip():  # Only keep non-empty texts
            new_texts.append(text)
    
    if new_texts:
        st.session_state.all_texts.extend(new_texts)
        st.session_state.store = create_index(st.session_state.all_texts, model)
        st.success(f"{len(new_texts)} new files processed and added to knowledge base!")
    else:
        st.warning("No readable text found in uploaded files.")

# Ask questions
if st.session_state.store:
    query = st.text_input("Ask a question about your documents:")
    if query and st.button("Search"):
        with st.spinner("Searching relevant content..."):
            contexts = retrieve(query, st.session_state.store, model)
            st.subheader("🔍 Retrieved Contexts")
            for i, ctx in enumerate(contexts, 1):
                st.markdown(f"**Context {i}:** {ctx[:400]}...")

        with st.spinner("Synthesizing answer..."):
            answer = synthesize_answer(query, contexts)
            st.subheader("🧠 Synthesized Answer")
            st.write(answer)
else:
    st.info("Please upload at least one file to start.")
