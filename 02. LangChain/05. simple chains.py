import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# --- 2. Inicializando o Modelo de Linguagem (LLM) ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, api_key=api_key)

# # --- 3. Criando o Template de Prompt Simples ---
# promp_template = PromptTemplate.from_template("Me fale sobre o carro {carro}.")

# # --- 4. Criando a Cadeia Simples ---
# runnable_sequence = promp_template | llm | StrOutputParser()

# # --- 5. Gerando a Resposta ---
# response = runnable_sequence.invoke({"carro": "HRV 2024"})
# print(response)

# --- 3. Criando o Template de Prompt Simples com nova sintaxe---
runnable_sequence = (
    PromptTemplate.from_template("Me fale sobre o carro {carro}.")
    | llm
    | StrOutputParser()
)
# --- 4. Gerando a Resposta ---
response = runnable_sequence.invoke({"carro": "HRV 2024"})
print(response)