import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, api_key=api_key)

wikipedia_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        lang="pt", 
        extract_format="html",
    )
)
# Create an agent that can query Wikipedia and summarize the results
agent_executor = create_python_agent(
    llm=llm,
    tool=wikipedia_tool,
    verbose=True,
)

# Define a prompt template for the agent
prompt_template = PromptTemplate(
    input_variables=["query"],
    template="Pesquise na web sobre {query} e forneça um resumo sobre o assunto",
)
# Create a prompt template for the agent to use
query = "Alan Turing"
prompt = prompt_template.format(query=query)
response = agent_executor.invoke(prompt)
print(response["output"])