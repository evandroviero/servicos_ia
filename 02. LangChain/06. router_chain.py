import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, api_key=api_key)


classification_chain = (
    PromptTemplate.from_template(
        """
        Classifique a pergunta do usuário em uma dos seguintes setores:
        - Financeiro
        - Suporte Técnico
        - Outras informações
        Pergunta: {pergunta}
        """
    )
    | llm
    | StrOutputParser()
)

finacial_chain = (
    PromptTemplate.from_template(
        """
        Você é um especialista financeiro.
        Sempre responda às perguntas começando com 'Bem-vindo ao Setor Financeiro'.
        Responda à pergunta do usuário:
        Pergunta: {pergunta}
        """
    )
    | llm
    | StrOutputParser()
)

tech_support_chain = (
    PromptTemplate.from_template(
        """
        Você é um especialista em suporte técnico.
        Sempre responda às perguntas começando com 'Bem-vindo ao setor de Suporte Técnico'.
        Responda à pergunta do usuário:
        Pergunta: {pergunta}
        """
    )
    | llm
    | StrOutputParser()
)

other_info_chain = (
    PromptTemplate.from_template(
        """
        Você é um especialista em informações gerais.
        Sempre responda às perguntas começando com 'Bem-vindo ao setor de Outras Informações'.
        Responda à pergunta do usuário:
        Pergunta: {pergunta}
        """
    )
    | llm
    | StrOutputParser()
)


def route(classification):
    if 'financeiro' in classification.lower():
        return finacial_chain
    elif 'técnico' in classification.lower():
        return tech_support_chain
    else:
        return other_info_chain  
    
pergunta = "Quando vai vencer o meu boleto?"

classification = classification_chain.invoke({"pergunta": pergunta})
response_chain = route(classification=classification)
response = response_chain.invoke({"pergunta": pergunta})

print(f"Pergunta: {pergunta}")
print(f"Classificação: {classification}")
print(f"Resposta: {response}")