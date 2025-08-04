import re
import pytesseract
import streamlit as st
from PIL import Image
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
os.environ["ALLOW_RESET"] = "TRUE"


def get_text(file):
    image = Image.open(file)
    text = pytesseract.image_to_string(image, lang='eng')
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GEMINI_API_KEY")) 
    PineconeVectorStore.from_texts(chunks, embeddings, index_name=os.getenv("PINECONE_INDEX_NAME_IMAGES"))
    vstore = PineconeVectorStore(embedding=embeddings, index_name=os.getenv("PINECONE_INDEX_NAME_IMAGES"))
    return vstore

def user_input(user_question, vstore):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.3,google_api_key=os.getenv("GEMINI_API_KEY"))
    retrieval_chat_prompt = """
    You are a helpful assistant that answers questions about images.
    The text can be jumbled or not clear so you can assume some data ** provided ** that if you have enough useful context.
    If you can't find an answer, just say that you don't know.

    Context: {context}
    Question: {question}"""

    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=retrieval_chat_prompt)
    retrieval_chain = create_retrieval_chain(vstore.as_retriever(), combine_docs_chain)
    response = retrieval_chain.invoke({"input": user_question})
    st.write(response["answer"])

def main():
    st.set_page_config("Chat with images")
    file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
    if st.button("Submit & Process"): 
        if file is None:
            st.error("Please upload an image first")
            return
        with st.spinner("Processing..."):
            text = get_text(file)
            text_chunks = get_text_chunks(text)
            vstore = get_vector_store(text_chunks)
            st.session_state.vstore = vstore
            st.success("Done")

    st.header("Chat with images")
    user_question = st.text_input("Ask a Question from the Image")
    if st.button("Submit"):
        if user_question:
            if "vstore" not in st.session_state or st.session_state.vstore is None:
                st.error("Please upload an image first")
                return
            else:
                user_input(user_question, st.session_state.vstore)


if __name__ == "__main__": 
    main()