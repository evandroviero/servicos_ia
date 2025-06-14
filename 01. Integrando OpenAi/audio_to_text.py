from openai import OpenAI
import os
from dotenv import load_dotenv


load_dotenv()

client = OpenAI(
  api_key= os.getenv("OPENAI_API_KEY"),
)

audio_file = open("audio.mp3", "rb")
response = client.audio.transcriptions.create(
  file=audio_file,
  model="whisper-1",
  response_format="text",
  language="pt",
  prompt="",
  temperature=0.0,
  max_tokens=1000,
  completion_params={
    "stop": ["\n"],
  },
)
print(response.text)
audio_file.close()


with open("transcription.txt", "w") as f:
    f.write(response.text)
           