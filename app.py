import os
import io
import faiss
import pickle
import streamlit as st
from PIL import Image
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from typing import List
from openai import OpenAI
import easyocr  # for OCR on images

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="RAG Q&A App", page_icon="📚")

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")))

EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_FILE = "uploaded_vector_store.pkl"

# ----------------------------
# UTILS
# ----------------------------
def extract_text_from_pdf(file):
    """Extracts text from a PDF file"""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()

def extract_text_from_image(file):
    """Extract text robustly from any uploaded image"""
    try:
        image = Image.open(file).convert("RGB")
        image_np = np.array(image)
        reader = easyocr.Reader(["en"], verbose=False)
        results = reader.readtext(image_np, detail=0)
        return " ".join(results).strip() or "[No readable text found]"
    except Exception as e:
        return f"[Error reading image: {str(e)}]"


def split_text(text: str, chunk_size=500, overlap=100):
    """Splits long text into smaller chunks for embedding"""
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
def build_index_from_texts(texts: List[str]):
    """Creates FAISS index from text chunks"""
    model = SentenceTransformer(EMBED_MODEL)
    chunks = []
    for doc in texts:
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
    """Retrieves most relevant text chunks for a query"""
    q_emb = model.encode([query], convert_to_numpy=True)
    D, I = store["index"].search(q_emb, top_k)
    return [store["chunks"][i] for i in I[0]]

# ----------------------------
# SYNTHESIS
# ----------------------------
def synthesize_answer(query, contexts):
    """Uses OpenAI to synthesize a concise answer from context"""
    context_text = "\n\n".join(contexts)
    prompt = f"""
    You are a helpful assistant. Use the provided context to answer the user's question.
    If the answer is not in the context, say you don't know.

    Context:
    {context_text}

    Question: {query}
    Answer:
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# ----------------------------
# STREAMLIT UI
# ----------------------------
def main():
    st.title("📚 Knowledge-base Q&A (PDF + Image)")
    st.markdown("Upload a **PDF** or **Image**, then ask questions about its content!")

    uploaded_files = st.file_uploader(
        "Upload files (PDF or Image)", 
        type=["pdf", "png", "jpg", "jpeg"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        all_texts = []
        for file in uploaded_files:
            if file.type == "application/pdf":
                text = extract_text_from_pdf(file)
                st.success(f"✅ Extracted text from PDF: {file.name}")
            else:
                text = extract_text_from_image(file)
                st.write("✅ Extracted text preview:", text[:1000])

                st.success(f"✅ Extracted text from Image: {file.name}")
            all_texts.append(text)

        if st.button("🔍 Build Knowledge Base"):
            with st.spinner("Creating embeddings and index..."):
                store, model = build_index_from_texts(all_texts)
                st.success("✅ Vector store created successfully!")

            query = st.text_input("Ask a question about your uploaded content:")
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
        st.info("👆 Please upload at least one PDF or image file to begin.")

if __name__ == "__main__":
    main()
