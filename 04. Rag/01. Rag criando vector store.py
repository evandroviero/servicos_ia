import os
from dotenv import load_dotenv
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, api_key=api_key)

path = "data/laptop_manual.pdf"
loarder = PyPDFLoader(path)
docs = loarder.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=600,
)
chunks = text_splitter.split_documents(
    documents=docs
)

persist_directory = "db"

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vecttor_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="db",
    collection_name="manual_laptop",
)

retriever = vecttor_store.as_retriever()

prompt = hub.pull("rlm/rag-prompt")

rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

try:
    while True:
        question = input("Ask a question: ")
        if question.lower() in ["exit", "quit"]:
            break
        response = rag_chain.invoke(question)
        print(f"Answer: {response}")
except KeyboardInterrupt:
    print("\nExiting...")
    exit()