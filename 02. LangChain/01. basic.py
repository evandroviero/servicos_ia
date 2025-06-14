import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    client=os.getenv("GOOGLE_GENAI_API_KEY"),
)

messages = [
    {"role": "system", "content": "Você é um assistente inteligente que responde perguntas sobre ciência da computação."},
    {"role": "user", "content": "Quem foi Alan Turing?"}
    ]
response = model.invoke(messages)
print(response)