import os
from dotenv import load_dotenv
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

class CreateCollection():
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.path = "data/carros.csv"
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.model = "gemini-1.5-flash-latest"
        self.llm = ChatGoogleGenerativeAI(model=self.model, temperature=0, api_key=self.api_key)

    def run(self):
        loader = CSVLoader(self.path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=600,
        )
        chunks = text_splitter.split_documents(documents=docs)

        persist_directory = "db"

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=self.collection_name,
        )
        return vector_store

collection = CreateCollection(collection_name="carros")
vector_store = collection.run()

api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, api_key=api_key)

persist_directory = "db"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector_store = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
    collection_name="carros",
)

retriever = vector_store.as_retriever()

system_prompt = """
Use o contexto para responder as perguntas.
Contexto: {context}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt,
)

chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=question_answer_chain,
)

query = "Qual o óleo devo usar para cada carro?"

response = chain.invoke({"input": query})
print(f"Resposta: {response.get('answer')}")