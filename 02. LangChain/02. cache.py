import os
import time  # Importado para demonstrar o cache
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
# --- 1. Importações para o Cache ---
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

# --- 2. Configuração do Ambiente ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# --- 3. Habilitando o Cache Globalmente ---
# Isso fará com que todas as chamadas de LLM neste script
# usem este cache em memória.
print("Configurando o cache em memória (InMemoryCache)...")
set_llm_cache(InMemoryCache())

# --- 4. Definindo o Modelo de Linguagem (LLM) ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, api_key=api_key)

# --- 5. Definindo as Mensagens com Roles ---
messages = [
    SystemMessage(content="Você é um assistente inteligente que responde perguntas sobre ciência da computação."),
    HumanMessage(content="Quem foi Alan Turing?")
]

# --- 6. Invocando o LLM com as Mensagens (Primeira Chamada) ---
print("\n--- [PRIMEIRA CHAMADA] Invocando o LLM... ---")
start_time = time.time()

try:
    response = llm.invoke(messages)
    print(response.content)

except Exception as e:
    print(f"Ocorreu um erro ao chamar o LLM: {e}")
    print("Verifique se sua GOOGLE_API_KEY está correta e se você tem acesso ao modelo.")

end_time = time.time()
print(f"--- Tempo da primeira chamada: {end_time - start_time:.2f} segundos ---\n")


# --- 7. Invocando o LLM Novamente (Demonstração do Cache) ---
# A mesma pergunta será feita. Desta vez, a resposta deve vir do cache.
print("\n--- [SEGUNDA CHAMADA] Invocando o LLM novamente com a mesma pergunta... ---")
start_time = time.time()

try:
    # A resposta virá do cache, pois a entrada (messages) é idêntica.
    cached_response = llm.invoke(messages)
    print(cached_response.content)

except Exception as e:
    print(f"Ocorreu um erro ao chamar o LLM: {e}")

end_time = time.time()
print(f"--- Tempo da segunda chamada (em cache): {end_time - start_time:.4f} segundos ---")
print("\nObserve como a segunda chamada foi praticamente instantânea.")