import os
import json
import sqlite3
import streamlit as st
import openai
import PyPDF2
import numpy as np
# ---- Config ----
# Set your Azure OpenAI values here or as environment variables.
# https://innovate-openai-api-mgt.azure-api.net/innovate-tracked/deployments/{deployment-id}/chat/completions?api-version={api-version}
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "85015946c55b4763bcc88fc4db9071dd")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://innovate-openai-api-mgt.azure-api.net/innovate-tracked/deployments/{deployment-id}/chat/completions?api-version=2024-02-01")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "ada-002")
AZURE_OPENAI_COMPLETION_DEPLOYMENT = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT", "gpt-4o-mini")
# Configure the OpenAI client for Azure
openai.api_type = "azure"
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_version = "2024-02-01"  # use the version required by your deployment
openai.api_key = AZURE_OPENAI_KEY
DB_PATH = "vector_store.db"
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
   response = openai.Embedding.create(
       input=text,
       deployment_id=AZURE_OPENAI_EMBEDDING_DEPLOYMENT
   )
   embedding = response['data'][0]['embedding']
   return embedding
def cosine_similarity(vec1: list, vec2: list) -> float:
   """Computes the cosine similarity between two vectors."""
   vec1 = np.array(vec1)
   vec2 = np.array(vec2)
   return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
# ---- Database Functions ----
def init_db():
   """Initializes the SQLite database and creates the table if it does not exist."""
   conn = sqlite3.connect(DB_PATH)
   cursor = conn.cursor()
   cursor.execute("""
       CREATE TABLE IF NOT EXISTS pdf_chunks (
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           pdf_name TEXT,
           chunk_text TEXT,
           embedding TEXT
       )
   """)
   conn.commit()
   conn.close()
def add_chunk(pdf_name: str, chunk_text: str, embedding: list):
   """Inserts a PDF chunk and its embedding into the database."""
   conn = sqlite3.connect(DB_PATH)
   cursor = conn.cursor()
   cursor.execute(
       "INSERT INTO pdf_chunks (pdf_name, chunk_text, embedding) VALUES (?, ?, ?)",
       (pdf_name, chunk_text, json.dumps(embedding))
   )
   conn.commit()
   conn.close()
def search_chunks(query_embedding: list, top_k: int = 3) -> list:
   """Searches the database for the top_k chunks most similar to the query embedding."""
   conn = sqlite3.connect(DB_PATH)
   cursor = conn.cursor()
   cursor.execute("SELECT chunk_text, embedding FROM pdf_chunks")
   rows = cursor.fetchall()
   conn.close()
   similarities = []
   for chunk_text, embedding_str in rows:
       embedding = json.loads(embedding_str)
       sim = cosine_similarity(query_embedding, embedding)
       similarities.append((chunk_text, sim))
   similarities.sort(key=lambda x: x[1], reverse=True)
   top_chunks = [chunk for chunk, sim in similarities[:top_k]]
   return top_chunks
def answer_query(question: str) -> str:
   """Retrieves context based on the question and queries the LLM for an answer."""
   # Get embedding for the question using Azure OpenAI
   question_embedding = get_embedding(question)
   relevant_chunks = search_chunks(question_embedding)
   context = "\n\n".join(relevant_chunks)
   prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
   response = openai.ChatCompletion.create(
       deployment_id=AZURE_OPENAI_COMPLETION_DEPLOYMENT,
       messages=[
           {"role": "system", "content": "You are an expert assistant."},
           {"role": "user", "content": prompt}
       ],
       temperature=0.7,
   )
   answer = response["choices"][0]["message"]["content"]
   return answer
# ---- Streamlit UI ----
st.title("PDF QA Application (POC with Azure OpenAI)")
# Initialize the database on first run
init_db()
st.header("Upload a PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
   st.write("Processing PDF...")
   text = extract_text_from_pdf(uploaded_file)
   if text:
       chunks = chunk_text(text)
       for chunk in chunks:
           embedding = get_embedding(chunk)
           add_chunk(uploaded_file.name, chunk, embedding)
       st.success("PDF processed and embeddings stored locally.")
   else:
       st.error("No text could be extracted from the PDF.")
st.header("Ask a Question")
question = st.text_input("Enter your question:")
if st.button("Get Answer") and question:
   st.write("Querying the model...")
   answer = answer_query(question)
   st.subheader("Answer:")
   st.write(answer)
