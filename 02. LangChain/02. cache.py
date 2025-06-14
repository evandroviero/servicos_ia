import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.cache import InMemoryCache, SQLiteCache
from langchain.globals import set_llm_cache

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    client=os.getenv("GOOGLE_GENAI_API_KEY"),
)

# set_llm_cache(InMemoryCache()) # apenas localmente, não persiste entre reinicializações
set_llm_cache(SQLiteCache(database_path="langchain_cache.db"))

messages = [
    {"role": "system", "content": "Você é um assistente inteligente que responde perguntas sobre ciência da computação."},
    {"role": "user", "content": "Quem foi Alan Turing?"}
]

response_1 = model.invoke(messages)
response_2 = model.invoke(messages)

print("Response 1:", response_1)
print("Response 2:", response_2)


