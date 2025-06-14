from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
  api_key= os.getenv("OPENAI_API_KEY"),
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "Me fale mais sobre nossa conversa"}
  ]
)

print(completion.choices[0].message.content)

stream = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "Me fale mais sobre nossa conversa"}
  ],
  stream=True # Resposta igual chatgpt 
  temperature=0.2 # Temperatura é a aleatoriedade da resposta, quanto menor mais previsível será a resposta
  max_tokens=1000, # Limite de tokens da resposta
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
print("\n\nStream finalizado")


# Usando role com system para definir o contexto da conversa
response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
      {"role": "system", "content": "Você é um auditor médico. De respostas sobre validações de glosas."},
      {"role": "user", "content": "Posso cobrar uma consulta médica realizada por um profissional de saúde que não é médico?"}
  ]
)



