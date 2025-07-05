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

def ask_question(model, query, vector_store):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, api_key=os.getenv("GOOGLE_API_KEY"))
    retriever = vector_store.as_retriever()

    system_prompt = """"
    Use o contexto para responder as perguntas.
    Se não encontrar uma resposta no contexto,
    explique que não há informaçãoes disponíveis.
    Responda em formato de markdown e com visualizações
    elaboradas e interativas.
    Contexto: {context}
    """
    messages = [("system", system_prompt)]

    for message in st.session_state.messages:
        messages.append((message.get("role"), message.get("content")))
    
    messages.append(("human", "{input}"))

    prompt = ChatPromptTemplate.from_messages(messages)

    question_ansewer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )

    chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_ansewer_chain,
    )

    response = chain.invoke({"input": query})
    return response.get("answer")


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

if "messages" not in st.session_state:
    st.session_state["messages"] = []

question = st.chat_input("Como posso ajudar?")

if vector_store and question:
    
    for message in st.session_state.messages:
        st.chat_message(message.get("role")).write(message.get("content"))
    
    st.chat_message("user").write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Buscando resposta..."):
        response = ask_question(
            model= selected_model,
            query= question,
            vector_store=vector_store,
        )

        st.chat_message("ai").write(response)
        st.session_state.messages.append({"role": "ai", "content": response})