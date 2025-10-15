import os
import faiss
import pickle
import openai
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from typing import List


openai.api_key = os.getenv("OPENAI_API_KEY")  # or st.secrets["OPENAI_API_KEY"]
EMBED_MODEL = "all-MiniLM-L6-v2"


def read_pdf(file) -> str:
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def read_txt(file) -> str:
    return file.read().decode("utf-8")

def split_text(text: str, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


@st.cache_resource
def load_model():
    return SentenceTransformer(EMBED_MODEL)

def build_index(chunks, model):
    embeddings = model.encode(chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


def retrieve(query, store, model, top_k=3):
    q_emb = model.encode([query], convert_to_numpy=True)
    D, I = store["index"].search(q_emb, top_k)
    return [store["chunks"][i] for i in I[0]]

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


def main():
    st.title("📚 Dynamic Knowledge-base Search Engine (RAG Demo)")
    st.markdown("Upload your PDF or TXT files and start querying immediately.")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
    )

    if uploaded_files:
        model = load_model()
        all_chunks = []

        for file in uploaded_files:
            if file.name.endswith(".pdf"):
                text = read_pdf(file)
            else:
                text = read_txt(file)
            all_chunks.extend(split_text(text))

       
        index, embeddings = build_index(all_chunks, model)
        store = {"index": index, "chunks": all_chunks}

        st.success(f"{len(all_chunks)} chunks created from uploaded files!")

      
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
