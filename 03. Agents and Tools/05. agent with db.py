import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, api_key=api_key)

db = SQLDatabase.from_uri("sqlite:///data/ipca.db")

toolkit = SQLDatabaseToolkit(
    db=db, 
    llm=llm
)

system_message = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    tools=toolkit.get_tools(),
    prompt=system_message,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=toolkit.get_tools(),
    verbose=True,
)

prompt = """"
Use as ferramentas necessárias para responder às perguntas relacionadas ao histórico do IPCA ao longo do tempo.
Responda tudo em português brasileiro.
Se não souber a resposta, diga que não sabe.
Pergunta: {query}
"""

prompt_template = PromptTemplate.from_template(prompt)

question = "Baseado nos dados históricos do IPCA, usando matemática estatística, faça uma previsão dos valores de cada mês até o final de 2024."

output = agent_executor.invoke({
    "input": prompt_template.format(query=question)
})

print(output["output"])