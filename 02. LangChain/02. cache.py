import os
from getpass import getpass
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# --- 1. Configuração do Ambiente ---
# Certifique-se de ter sua chave de API do Google AI Studio configurada.
# Ela será solicitada se não estiver como variável de ambiente.
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass("Insira sua chave de API do Google AI: ")

# --- 2. Inicializando o Modelo de Linguagem (LLM) ---
# Usaremos o modelo "gemini-pro" para este exemplo.
# temperature=0 para respostas mais consistentes.
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

# --- 3. Definindo as Mensagens com Roles ---
# As mensagens são formatadas como objetos SystemMessage e HumanMessage do LangChain,
# que são compatíveis com os modelos de chat.
messages = [
    SystemMessage(content="Você é um assistente inteligente que responde perguntas sobre ciência da computação."),
    HumanMessage(content="Quem foi Alan Turing?")
]

# --- 4. Invocando o LLM com as Mensagens ---
print("--- Resposta do LLM ---")
try:
    # A função .invoke() envia as mensagens para o modelo e retorna a resposta.
    response = llm.invoke(messages)

    # A resposta do LLM é um objeto ChatMessage (normalmente AIMessage).
    # O conteúdo da resposta está no atributo .content.
    print(response.content)

except Exception as e:
    print(f"Ocorreu um erro ao chamar o LLM: {e}")
    print("Verifique se sua GOOGLE_API_KEY está correta e se você tem acesso ao modelo 'gemini-pro'.")