import os
from getpass import getpass
import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# --- 1. Configuração do Ambiente ---
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass("Enter your Google AI API key: ")

# --- 2. Definindo o Modelo de Linguagem (LLM) ---
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

# --- 3. Definindo as Ferramentas (Tools) ---
@tool
def get_current_datetime(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Returns the current date and time in the specified format."""
    return datetime.datetime.now().strftime(format)

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the result."""
    return a * b

tools = [get_current_datetime, multiply]

# --- 4. Criando o Agente ---

# Prompt base para o agente
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the tools provided to answer questions."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Criar o agente
agent = create_tool_calling_agent(llm, tools, prompt)

# Criar o AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 5. Executando o Agente ---
print("--- Executando o Agente ---")

# Exemplo 1: Usando get_current_datetime
print("\nPergunta 1: What is the current date and time?")
response1 = agent_executor.invoke({"input": "What is the current date and time?"})
print("Resposta 1:", response1["output"])

# Exemplo 2: Usando multiply
print("\nPergunta 2: What is 15 multiplied by 8?")
response2 = agent_executor.invoke({"input": "What is 15 multiplied by 8?"})
print("Resposta 2:", response2["output"])

# Exemplo 3: LLM responde diretamente
print("\nPergunta 3: Tell me a fun fact about giraffes.")
response3 = agent_executor.invoke({"input": "Tell me a fun fact about giraffes."})
print("Resposta 3:", response3["output"])

# Exemplo 4: Combinando o uso de ferramentas e raciocínio do LLM
print("\nPergunta 4: What is the current date and time, and then what is 20 multiplied by 5?")
response4 = agent_executor.invoke({"input": "What is the current date and time, and then what is 20 multiplied by 5?"})
print("Resposta 4:", response4["output"])