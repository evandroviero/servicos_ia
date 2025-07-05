import os
from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, api_key=api_key)

persist_directory = "db"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector_store = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
    collection_name="manual_laptop",
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

query = "Qual a marca e o modelo do notebook?"

response = chain.invoke({"input": query})
print(f"Resposta: {response.get('answer')}")