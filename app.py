import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import wikipedia

# Configure the Groq API key directly
api_key = "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"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    try:
        vector_store = FAISS.from_texts(text_chunks, embedding=None)  # No embeddings used here
        vector_store.save_local("faiss_index")
    except Exception as e:
        print(f"Error during embedding: {e}")
        st.error("An error occurred while embedding the documents. Please check the logs for more details.")

def get_conversational_chain(retriever):
    prompt_template = """
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGroq(api_key=api_key, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    return chain

def user_input(user_question):
    try:
        new_db = FAISS.load_local("faiss_index", embedding=None, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        # Retrieve information from Wikipedia
        wiki_summary = wikipedia.summary(user_question, sentences=2)
        docs.append({"text": wiki_summary})

        chain = get_conversational_chain(new_db.as_retriever())
        response = chain.invoke({"input_documents": docs, "question": user_question})

        print(response)
        st.write("Reply: ", response["output_text"])
    except Exception as e:
        print(f"Error during user input processing: {e}")
        st.error("An error occurred while processing your question. Please try again.")

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Groq💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
