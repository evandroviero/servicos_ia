from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
  api_key= os.getenv("OPENAI_API_KEY"),
)

response = client.audio.speech.create(
    model="tts-1",
    voice="nova",
    input="Olá, tudo bem? Eu sou um assistente virtual e estou aqui para ajudar você com suas dúvidas.",
)

response.write_to_file("01. Integrando OpenAi/audio.mp3")
