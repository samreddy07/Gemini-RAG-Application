from azure.ai.openai import OpenAIClient
from azure.ai.openai.models import EmbeddingRequest, ChatCompletionRequest
import streamlit as st
from faiss import FAISS

def get_conversational_chain():
    client = OpenAIClient(api_key="YOUR_AZURE_OPENAI_API_KEY")
    model = "YOUR_AZURE_OPENAI_MODEL_NAME"
    temperature = 0.3

    prompt_template = "YOUR_PROMPT_TEMPLATE"
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(client, model=model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    client = OpenAIClient(api_key="YOUR_AZURE_OPENAI_API_KEY")
    embeddings = client.embeddings(EmbeddingRequest(model="YOUR_AZURE_OPENAI_EMBEDDING_MODEL", input=user_question))

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Azure OpenAI💁")

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
