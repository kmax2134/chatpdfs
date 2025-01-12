import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Hugging Face API token
HF_TOKEN = os.getenv("HF_TOKEN")  # Add your Hugging Face API token in the .env file

# Configure Groq API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Ensure the Groq API key is stored in .env
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found. Please set it in your environment variables.")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please set it in your environment variables.")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

from langchain.embeddings import HuggingFaceEmbeddings

def get_vector_store(text_chunks):
    # Initialize Hugging Face Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        api_key=HF_TOKEN  # Ensure this is properly set
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    
    # Initialize ChatGroq with the Groq API Key
    model = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.5, api_key=GROQ_API_KEY)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    # Reuse Hugging Face Embeddings
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_TOKEN,
        model_name="BAAI/bge-base-en-v1.5"
    )
    
    # Load FAISS vector store
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    # Get conversational chain
    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Multi PDF Chatbot", page_icon=":scroll:")
    st.header("Multi-PDF's 📚 - Chat Agent 🤖 ")

    user_question = st.text_input("Ask a Question from the PDF Files uploaded .. ✍️📝")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.image("img/Robot.jpg")
        st.write("---")
        
        st.title("📁 PDF File's Section")
        pdf_docs = st.file_uploader("Upload your PDF Files & \n Click on the Submit & Process Button ", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):  # User-friendly message.
                raw_text = get_pdf_text(pdf_docs)  # Get the PDF text
                text_chunks = get_text_chunks(raw_text)  # Get the text chunks
                get_vector_store(text_chunks)  # Create vector store
                st.success("Done")
        
        st.write("---")
        st.image("img/gkj.jpg")
        st.write("AI App created by @ Gurpreet Kaur")  # Add this line to display the image

    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 15px; text-align: center;">
            © <a href="https://github.com/gurpreetkaurjethra" target="_blank">Gurpreet Kaur Jethra</a> | Made with ❤️
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
