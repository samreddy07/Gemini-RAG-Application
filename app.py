import streamlit as st
import faiss
import numpy as np
import json
import os
import PyPDF2
from openai import AzureOpenAI  # Import AzureOpenAI SDK
# === CONFIGURATION ===
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "85015946c55b4763bcc88fc4db9071dd")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://innovate-openai-api-mgt.azure-api.net/innovate-tracked/deployments/gpt-4o-mini/chat/completions?api-version=2024-02-01")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "ada-002")
AZURE_OPENAI_COMPLETION_DEPLOYMENT = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT", "gpt-4o-mini")
FAISS_INDEX_PATH = "faiss_index.bin"
FAISS_METADATA_PATH = "faiss_metadata.json"
# Initialize Azure OpenAI client
client = AzureOpenAI(
   api_key=AZURE_OPENAI_KEY,
   api_version ="2024-02-01",
   azure_endpoint = "https://innovate-openai-api-mgt.azure-api.net/innovate-tracked/deployments/ada-002/embeddings?api-version=2024-02-01"
)

chat_client = AzureOpenAI(
   api_key=AZURE_OPENAI_KEY,
   api_version ="2024-02-01",
   azure_endpoint = "https://innovate-openai-api-mgt.azure-api.net/innovate-tracked/deployments/gpt-4o-mini/chat/completions?api-version=2024-02-01"
)
# === FAISS STORE ===
class FAISSStore:
   def __init__(self, embedding_dim=1536):
       self.embedding_dim = embedding_dim
       self.index = faiss.IndexFlatL2(self.embedding_dim)  # L2 distance search
       self.metadata = []  # Stores text chunks
       # Load existing index if available
       if os.path.exists(FAISS_INDEX_PATH):
           self.load_index()
   def add_embeddings(self, texts, embeddings):
       """Adds text chunks & embeddings to FAISS"""
       if not embeddings:
           return
       vectors = np.array(embeddings).astype("float32")
       self.index.add(vectors)
       self.metadata.extend(texts)
       self.save_index()
   def search(self, query_embedding, top_k=3):
       """Searches FAISS for the closest text chunks"""
       query_vector = np.array(query_embedding).astype("float32").reshape(1, -1)
       distances, indices = self.index.search(query_vector, top_k)
       results = [self.metadata[idx] for idx in indices[0] if idx < len(self.metadata)]
       return results
   def save_index(self):
       """Save FAISS index & metadata to disk"""
       faiss.write_index(self.index, FAISS_INDEX_PATH)
       with open(FAISS_METADATA_PATH, "w") as f:
           json.dump(self.metadata, f)
   def load_index(self):
       """Load FAISS index & metadata from disk"""
       self.index = faiss.read_index(FAISS_INDEX_PATH)
       with open(FAISS_METADATA_PATH, "r") as f:
           self.metadata = json.load(f)
# Initialize FAISS
faiss_store = FAISSStore()
# === PDF PROCESSOR ===
def extract_text_from_pdf(pdf_file):
   """Extracts text from an uploaded PDF file."""
   text = ""
   pdf_reader = PyPDF2.PdfReader(pdf_file)
   for page in pdf_reader.pages:
       text += page.extract_text() + "\n"
   return text
# === EMBEDDING FUNCTION ===
def get_embedding(text):
   """Generates embeddings using Azure OpenAI."""
   response = client.embeddings.create(
       input=text,
       model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT
   )
   return response.data[0].embedding
# === STREAMLIT UI ===
st.set_page_config(page_title="PDF Q&A Bot", layout="wide")
st.title("📄 PDF Q&A Chatbot using FAISS and Azure OpenAI")
# 📂 PDF Upload
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
   with st.spinner("🔍 Processing PDF..."):
       text = extract_text_from_pdf(uploaded_file)
       chunks = text.split("\n")  # Simple chunking (can be improved)
       embeddings = [get_embedding(chunk) for chunk in chunks if chunk.strip()]
       faiss_store.add_embeddings(chunks, embeddings)
       st.success("✅ PDF processed and stored in FAISS!")
# 💬 Chatbot Input
query = st.text_input("Ask a question:")
if query:
   with st.spinner("🤖 Fetching answer..."):
       query_embedding = get_embedding(query)
       relevant_chunks = faiss_store.search(query_embedding, top_k=3)
       context = "\n".join(relevant_chunks)
       messages = [
           {"role": "system", "content": "You are a helpful assistant."},
           {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
       ]
       response = chat_client.chat.completions.create(
           model=AZURE_OPENAI_COMPLETION_DEPLOYMENT,
           messages=messages,
           temperature=0.3,
       )
       answer = response.choices[0].message.content
       st.markdown("### 📜 Answer:")
       st.success(answer)
