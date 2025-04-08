import streamlit as st
from llama_index.core import (
   SimpleDirectoryReader,
   VectorStoreIndex,
   StorageContext)
from llama_index.core.vector_stores import (
   MetadataFilters,
   FilterCondition,
   ExactMatchFilter
)
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.core.embeddings import resolve_embed_model
import chromadb
import os
from pathlib import Path
# === Azure OpenAI CONFIGURATION ===
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "85015946c55b4763bcc88fc4db9071dd")
# Your Azure endpoint should be the base URL for the Azure OpenAI resource.
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://innovate-openai-api-mgt.azure-api.net")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
st.title(":underage: Mr. Wiki")
# Create documents directory if it doesn't exist
if "file_dir" not in st.session_state:
   st.session_state.file_dir = Path("./documents").resolve()
   os.makedirs(st.session_state.file_dir, exist_ok=True)
# Set up the vector database and options
if "options" not in st.session_state:
   db = chromadb.PersistentClient(path="./chroma_db")
   st.session_state.collection = db.get_or_create_collection("quickstart")
   st.session_state.vector_store = ChromaVectorStore(chroma_collection=st.session_state.collection)
   data = st.session_state.collection.get(include=['metadatas'])
   if data is not None:
       file_names = set(Path(i['file_name']).name for i in data['metadatas'])
       file_names = sorted(list(set(file_names)))
   else:
       file_names = []
   st.session_state.options = file_names
# Documents that we are going to query
if "selected_options" not in st.session_state:
   st.session_state.selected_options = []
# Initialize the LLM using Azure OpenAI credentials.
# This instance uses the Azure endpoint, key, and a specified deployment.
if "llm" not in st.session_state:
   st.session_state.llm = OpenAI(
       api_key=AZURE_OPENAI_KEY,
       api_base=AZURE_OPENAI_ENDPOINT,
       deployment_id=AZURE_OPENAI_DEPLOYMENT,
       temperature=0.3
   )
# Choose/resolve an embed model (this remains unchanged)
if "embed_model" not in st.session_state:
   st.session_state.embed_model = resolve_embed_model("local:BAAI/bge-small-zh-v1.5")
# If no documents are loaded, default to a simple chat engine.
if "chat_engine" not in st.session_state:
   st.session_state.chat_engine = SimpleChatEngine.from_defaults(llm=st.session_state.llm)
# --- CALLBACK FUNCTIONS ---
# Load data by filtering the documents from the vector store.
def load_data():
   with st.spinner(text="Loading Database..."):
       index = VectorStoreIndex.from_vector_store(
           st.session_state.vector_store, embed_model=st.session_state.embed_model
       )
       # IN operator is not supported by llama_index yet; use a set of OR filters instead.
       files = [str(st.session_state.file_dir / file) for file in st.session_state.selected_options]
       filters = [ExactMatchFilter(key="file_name", value=file) for file in files]
       meta_filters = MetadataFilters(
           filters=filters,
           condition=FilterCondition.OR
       )
       st.session_state.chat_engine = index.as_query_engine(
           verbose=True,
           llm=st.session_state.llm,
           streaming=True,
           filters=meta_filters
       )
# Delete files from both the vector store (Chroma collection) and the documents directory.
def delete(files):
   files = [str(st.session_state.file_dir / file) for file in files]
   with st.spinner(text="Deleting..."):
       st.session_state.collection.delete(where={"file_name": {"$in": files}})
       for file in files:
           if os.path.isfile(file):
               os.remove(file)
           st.session_state.options.remove(str(Path(file).name))
# Upload files into the database.
def upload(files):
   with st.spinner(text="Uploading..."):
       file_pathes = []
       for uploaded_file in files:
           if uploaded_file.name not in st.session_state.options:
               file_path = st.session_state.file_dir / uploaded_file.name
               bytes_data = uploaded_file.read()
               # Save the file to the documents directory.
               with open(file_path, "wb") as f:
                   f.write(bytes_data)
               st.session_state.options.append(uploaded_file.name)
               file_pathes.append(file_path)
       # Save the documents to the database.
       if file_pathes:
           documents = SimpleDirectoryReader(input_files=file_pathes).load_data()
           storage_context = StorageContext.from_defaults(vector_store=st.session_state.vector_store)
           VectorStoreIndex.from_documents(
               documents, storage_context=storage_context, embed_model=st.session_state.embed_model
           )
# --- SIDEBAR ELEMENTS ---
with st.sidebar:
   st.subheader(":desktop_computer: Vector Database Configuration")
   with st.expander(":arrow_up: Upload documents"):
       uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True)
       if uploaded_files:
           upload(uploaded_files)
   with st.expander(":wastebasket: Remove documents"):
       to_remove = st.multiselect(
           label="Delete documents",
           options=st.session_state.options
       )
       st.button(":fire: Confirm", on_click=lambda: delete(to_remove))
   st.markdown("---")
   st.subheader(":file_folder: Select documents needed to query")
   st.multiselect(
       label="Select documents needed to query",
       options=st.session_state.options,
       key="selected_options"
   )
   st.write(":bookmark_tabs: Selected documents are:", st.session_state.selected_options)
   st.button(":dancer: Query these documents!", on_click=load_data)
# --- MAIN CHAT PART ---
if "messages" not in st.session_state:
   st.session_state.messages = [
       {"role": "assistant", "content": "Ask me anything!"}
   ]
# Prompt for user input and add to the chat history.
if prompt := st.chat_input("Your question"):
   st.session_state.messages.append({"role": "user", "content": prompt})
# Display the chat history.
for message in st.session_state.messages:
   with st.chat_message(message["role"]):
       st.write(message["content"])
# Generate a new response if the last message is from the user.
if st.session_state.messages[-1]["role"] != "assistant":
   with st.chat_message("assistant"):
       with st.spinner("Thinking..."):
           full_response = []
           # Use streaming chat if supported; otherwise a simple query.
           if isinstance(st.session_state.chat_engine, SimpleChatEngine):
               stream = st.session_state.chat_engine.stream_chat(prompt)
           else:
               stream = st.session_state.chat_engine.query(prompt)
           # st.write_stream will handle streaming response content.
           response = st.write_stream(stream.response_gen)
           message = {"role": "assistant", "content": response}
           st.session_state.messages.append(message)
