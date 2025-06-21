import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.utilities import PythonREPL
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, api_key=api_key)

prompt = """"
Como assistente financeiro pessoal, que responderá as perguntas dando dicas financeiras e de investimentos.
Responda tudo em português brasileiro.
Pergunta: {query}
"""

prompt_template = PromptTemplate.from_template(prompt)

python_repl = PythonREPL()
python_repl_tool = Tool(
    name="Python REPL",
    description="""
    Um shell Python. Use isso para executar código Python. Execute apenas código Python válido.
    Se você precisar obter o retorno do código, use a função 'print(...)'. 
    Use para realizar cálculos financeiros necessários para responder as perguntas e dar dicas,
    """,
    func=python_repl.run,
)

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="DuckDuckGo Search",
    description="Útil para encontrar informações e dicas de economia e opções de investimento."
                "Você sempre deve pesquisar na internet as melhores dicas usando esta ferramenta, não"
                "responda diretamente. Sua resposta deve informar que há elementos pesquisados na internet.",
    func=search.run,
)

react_instructions = hub.pull("hwchase17/react")

tools = [python_repl_tool, search_tool]
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_instructions,
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

question = "Minha renda é de R$ 10000 por mês, o total de minhas despesas é de R$ 8500 mais R$ 1000 de alguel. Quais dicas de investimento você me dá?"

output = agent_executor.invoke(
    {"input": prompt_template.format(query=question)}
)
print(output['output'])