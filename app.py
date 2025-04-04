import streamlit as st
import faiss
import numpy as np
import json
import os
import PyPDF2
from openai import AzureOpenAI  
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
# === STREAMLIT UI SETTINGS ===
st.set_page_config(page_title="ChatGPT PDF Q&A", layout="wide")
st.title("📄 ChatGPT-Powered PDF Q&A Bot")
# Chat history storage
if "messages" not in st.session_state:
   st.session_state.messages = []
# === UI: PROFESSIONAL CHAT INTERFACE ===
st.markdown("""
<style>
   .message-container {
       max-width: 750px;
       padding: 10px 15px;
       margin: 5px 0;
       border-radius: 10px;
       font-size: 16px;
       display: inline-block;
       word-wrap: break-word;
   }
   .user-message {
       background-color: #DCF8C6;
       text-align: right;
       float: right;
       clear: both;
   }
   .bot-message {
       background-color: #F1F1F1;
       text-align: left;
       float: left;
       clear: both;
   }
   .chat-container {
       overflow-y: auto;
       height: 500px;
       border: 1px solid #ccc;
       padding: 10px;
       border-radius: 5px;
       background-color: #FFF;
   }
</style>
""", unsafe_allow_html=True)
# Display Chat History
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for message in st.session_state.messages:
   if message["role"] == "user":
       st.markdown(f"<div class='message-container user-message'>{message['content']}</div>", unsafe_allow_html=True)
   else:
       st.markdown(f"<div class='message-container bot-message'>{message['content']}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
# === PDF UPLOAD SECTION ===
uploaded_file = st.file_uploader("📂 Upload a PDF", type=["pdf"])
if uploaded_file:
   with st.spinner("🔍 Processing PDF..."):
       text = "\n".join([page.extract_text() for page in PyPDF2.PdfReader(uploaded_file).pages if page.extract_text()])
       chunks = text.split("\n")  
       embeddings = [client.embeddings.create(input=chunk, model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT).data[0].embedding for chunk in chunks if chunk.strip()]
       faiss_index = faiss.IndexFlatL2(1536)  
       vectors = np.array(embeddings).astype("float32")
       faiss_index.add(vectors)
       st.success("✅ PDF processed and stored in FAISS!")
# === USER QUESTION INPUT ===
query = st.text_input("💬 Type your question here:")
if query:
   with st.spinner("🤖 Fetching answer..."):
       query_embedding = client.embeddings.create(input=query, model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT).data[0].embedding
       relevant_chunks = [chunks[idx] for idx in faiss_index.search(np.array(query_embedding).reshape(1, -1), 3)[1][0]]
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
       # Store chat history
       st.session_state.messages.append({"role": "user", "content": query})
       st.session_state.messages.append({"role": "bot", "content": answer})
       # Refresh UI
       st.rerun()
