import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# --- 2. Inicializando o Modelo de Linguagem (LLM) ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, api_key=api_key)

# --- 3. Criando o Template de Prompt de Chat ---
chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content= "Você deve responder baseado em dados geográficos de regiões do Brasil."
        ),
        HumanMessagePromptTemplate.from_template("Por favor, me fale sobre a região {regiao} do Brasil."),
        AIMessage(content="Claro! Vou começar coletando informações sobre a região {regiao} do Brasil. Por favor, aguarde um momento."),
        HumanMessage(content="Certifique-se de incluir dados demográficos."),
        AIMessage(content="Entendido! Vou incluir dados:"),
    ]
)

prompt = chat_template.format_prompt(regiao="Sul")
# --- 4. Gerando a Resposta ---
response = llm.invoke(prompt.to_messages())
print(response.content)