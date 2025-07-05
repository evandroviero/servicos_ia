import os
import tempfile
from dotenv import load_dotenv
import streamlit as st

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, api_key=api_key)
persist_directory = "db"

def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    os.remove(temp_file_path)

    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 400
    )
    chunks = text_spliter.split_documents(documents=docs)
    return chunks

def load_existing_vector_store():
    if os.path.exists(os.path.join(persist_directory)):
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        )
        return vector_store
    return None

def add_to_vector_store(chuncks, vector_store=None):
    if vector_store:
        vector_store.add_documents(chuncks)
    else:
        vector_store = Chroma.from_documents(
            documents=chuncks,
            embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
            persist_directory=persist_directory,
        )
    return vector_store

vector_store = load_existing_vector_store()

st.set_page_config(
    page_title="Chat PyGPT",
    page_icon="📊"
)

st.header("🤖 Chat com seus documentos RAG")


with st.sidebar:
    st.header("Upload de arquivos")
    uploaded_files = st.file_uploader(
        label= "Faça o upload de arquivos PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        with st.spinner("Processando documentos..."):
            all_chunks = []
            for doc in uploaded_files:
                chunks = process_pdf(file=doc)
                all_chunks.extend(chunks)
            vector_store = add_to_vector_store(
                chuncks= all_chunks,
                vector_store = vector_store,
            )

    model_options =[
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4o-mini",
        "gpt-4o",
        "gemini-1.5-flash-latest",
    ]

    selected_model = st.sidebar.selectbox(
        label="Selecione o modelo LLM",
        options=model_options,
        index=model_options.index("gemini-1.5-flash-latest"),
        help="Selecione o modelo LLM que deseja usar"
    )

question = st.chat_input("Como posso ajudar?")

st.chat_message("user").write(question)
st.chat_message("ai").write("resposta IA")