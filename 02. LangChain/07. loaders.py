import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, api_key=api_key)

# loader = TextLoader("data/base_conhecimento.txt", encoding="utf-8")
# loader = PyPDFLoader("data/base_conhecimento.pdf")
loader = CSVLoader(file_path="data/base_conhecimento.csv", encoding="utf-8")
documents = loader.load()

prompt_base_conhecimento = PromptTemplate(
    input_variables=["contexto", "pergunta"],
    template="""Use o seguinte contexto para responder a pergunta:
    Responda apenas com base nas informações fornecidas.
    Não utilize informações externas ao contexto.
    Contexto: {contexto}
    Pergunta: {pergunta}
    """
)

chain = (
    prompt_base_conhecimento | llm | StrOutputParser()

)
response = chain.invoke(
    {
        "contexto": "\n".join(doc.page_content for doc in documents),
        "pergunta": "Qual óleo de motor devo usar?"
    }
)
print(response)

