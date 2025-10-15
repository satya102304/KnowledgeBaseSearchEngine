import os
import faiss
import pickle
import openai
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from typing import List

# ----------------------------
# CONFIG
# ----------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")  # or st.secrets["OPENAI_API_KEY"]
EMBED_MODEL = "all-MiniLM-L6-v2"
DATA_DIR = "docs"
INDEX_FILE = "vector_store.pkl"

# ----------------------------
# UTILS
# ----------------------------
def load_docs() -> List[str]:
    docs = []
    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)
        if file.endswith(".pdf"):
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            docs.append(text)
        elif file.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                docs.append(f.read())
    return docs

def split_text(text: str, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ----------------------------
# EMBEDDING + INDEX CREATION
# ----------------------------
@st.cache_resource
def build_or_load_index():
    model = SentenceTransformer(EMBED_MODEL)
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "rb") as f:
            store = pickle.load(f)
        return store, model

    docs = load_docs()
    chunks = []
    for doc in docs:
        chunks.extend(split_text(doc))

    embeddings = model.encode(chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    store = {"index": index, "chunks": chunks}
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(store, f)
    return store, model

# ----------------------------
# RETRIEVAL
# ----------------------------
def retrieve(query, store, model, top_k=3):
    q_emb = model.encode([query], convert_to_numpy=True)
    D, I = store["index"].search(q_emb, top_k)
    return [store["chunks"][i] for i in I[0]]

# ----------------------------
# SYNTHESIS
# ----------------------------
def synthesize_answer(query, contexts):
    context_text = "\n\n".join(contexts)
    prompt = f"Using the context below, answer the query concisely.\n\nContext:\n{context_text}\n\nQuestion: {query}\nAnswer:"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# ----------------------------
# STREAMLIT UI
# ----------------------------
def main():
    st.title("📚 Knowledge-base Search Engine (RAG Demo)")
    st.markdown("Upload your documents in the `docs/` folder and start querying.")

    store, model = build_or_load_index()

    query = st.text_input("Enter your query:")
    if st.button("Search"):
        if query:
            with st.spinner("Retrieving relevant info..."):
                contexts = retrieve(query, store, model)
                st.subheader("🔍 Retrieved Contexts")
                for i, ctx in enumerate(contexts, 1):
                    st.markdown(f"**Context {i}:** {ctx[:400]}...")

            with st.spinner("Synthesizing answer..."):
                answer = synthesize_answer(query, contexts)
                st.subheader("🧠 Synthesized Answer")
                st.write(answer)
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
