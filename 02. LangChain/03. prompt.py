import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    client=os.getenv("GOOGLE_GENAI_API_KEY"),
)

template = """
Traduza o texto do {idioma1} para o {idioma2}:
{texto}
"""

prompt_template = PromptTemplate.from_template(
    template=template
    )

prompt = prompt_template.format(
    idioma1="português",
    idioma2="inglês",
    texto="Olá, como você está?"
)
response = model.invoke(prompt)
print(response.content)
