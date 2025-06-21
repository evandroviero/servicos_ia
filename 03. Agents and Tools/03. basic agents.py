import os
from dotenv import load_dotenv
from langchain.agents import Tool
from langchain.prompts import PromptTemplate
from langchain_experimental.utilities import PythonREPL
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, api_key=api_key)

python_repl = PythonREPL()

python_repl_tool = Tool(
    name="Python REPL",
    description="""Um shell Python. Use isso para executar código Python. Execute apenas código Python válido.
    Se você precisar obter o retorno do código, use a função 'print(...)'.
    Use apra realizar cálculos financeiros necessários para responder as perguntas e dar dicas""",
    func=python_repl.run,
)

agent_executor = create_python_agent(
    llm=llm,
    tool=python_repl_tool,
    verbose=True,
    )

prompt_template = PromptTemplate(
    input_variables=["query"],
    template="""Resolva o cálculo {query}""",
)

query = "20 x 25"
prompt = prompt_template.format(query=query)
response = agent_executor.invoke(prompt)
print(f"Query: {query}")
print(f"Response: {response['output']}")