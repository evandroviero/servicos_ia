from openai import OpenAI
import os
from dotenv import load_dotenv
import requests

load_dotenv()

client = OpenAI(
  api_key= os.getenv("OPENAI_API_KEY"),
)

# usando modelos de imagem
image = client.images.generate(
    model="dall-e-3",
    prompt="Uma bela paisagem de montanhas ao amanhecer, com um lago refletindo as cores do céu.",
    n=1,
    quality='standard',
    size="1024x1024"
)
# Salvando a imagem gerada
image_url = image.data[0].url

response = requests.get(image_url)
if response.status_code == 200:
    with open("paisagem.png", "wb") as f:
        f.write(response.content)
    print("Imagem salva como paisagem.png")
else:
    print("Erro ao baixar a imagem:", response.status_code)