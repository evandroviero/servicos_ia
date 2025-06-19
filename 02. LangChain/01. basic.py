import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

# --- 1. Configuração do Ambiente ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# --- 2. Definindo o Modelo de Linguagem (LLM) ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, api_key=api_key)

# --- 3. Enviando uma Pergunta ao Modelo ---
pergunta = "Qual é a distância aproximada da Terra até a Lua em quilômetros?"
print(f"\nEnviando pergunta: {pergunta}")
resposta = llm.invoke(pergunta)
print("\nResposta do Gemini Flash:")
print(resposta.content)
