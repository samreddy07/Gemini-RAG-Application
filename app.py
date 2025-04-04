import os
import json
import streamlit as st
from openai import AzureOpenAI
import PyPDF2
import numpy as np
import faiss

# ---- Config ----
# Set your Azure OpenAI values here or as environment variables.
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "85015946c55b4763bcc88fc4db9071dd")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://innovate-openai-api-mgt.azure-api.net/innovate-tracked/deployments/ada-002/chat/completions?api-version=2024-02-01")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "ada-002")
AZURE_OPENAI_COMPLETION_DEPLOYMENT = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT", "gpt-4o-mini")

# Configure the OpenAI client for Azure
client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-02-01",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

chat_client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-02-01",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# ---- Utility Functions ----
def extract_text_from_pdf(file) -> str:
    """Extracts text from each page of an uploaded PDF file object."""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """Splits text into overlapping chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap  # overlapping window
    return chunks

def get_embedding(text: str) -> list:
    """Uses the Azure OpenAI API to get the embedding for the given text."""
    response = client.embeddings.create(
        input=text,
        model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT
    )
    embedding = response.data[0].embedding
    return embedding

# ---- FAISS Functions ----
def init_faiss_index(dimension: int):
    """Initializes the FAISS index."""
    index = faiss.IndexFlatL2(dimension)
    return index

def add_to_faiss_index(index, embeddings: list):
    """Adds embeddings to the FAISS index."""
    index.add(np.array(embeddings).astype('float32'))

def search_faiss_index(index, query_embedding: list, top_k: int = 3):
    """Searches the FAISS index for the top_k most similar embeddings."""
    distances, indices = index.search(np.array([query_embedding]).astype('float32'), top_k)
    return indices[0]

# ---- Streamlit UI ----
st.title("PDF QA Application (POC with Azure OpenAI)")

# Initialize the FAISS index on first run
dimension = 1536  # Example dimension, adjust based on your embedding size
faiss_index = init_faiss_index(dimension)
pdf_chunks = []

st.header("Upload a PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    st.write("Processing PDF...")
    text = extract_text_from_pdf(uploaded_file)
    if text:
        chunks = chunk_text(text)
        for chunk in chunks:
            embedding = get_embedding(chunk)
            pdf_chunks.append((uploaded_file.name, chunk, embedding))
            add_to_faiss_index(faiss_index, [embedding])
        st.success("PDF processed and embeddings stored in FAISS.")
    else:
        st.error("No text could be extracted from the PDF.")

st.header("Ask a Question")
question = st.text_input("Enter your question:")
if st.button("Get Answer") and question:
    st.write("Querying the model...")
    question_embedding = get_embedding(question)
    top_indices = search_faiss_index(faiss_index, question_embedding)
    relevant_chunks = [pdf_chunks[i][1] for i in top_indices]
    context = "\n\n".join(relevant_chunks)
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    response = chat_client.chat.completions.create(
        model=AZURE_OPENAI_COMPLETION_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are an expert assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )
    answer = response.choices[0].message.content
    st.subheader("Answer:")
    st.write(answer)
