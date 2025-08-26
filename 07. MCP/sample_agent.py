import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from tools import get_company_info, get_current_stock_price, get_history_stock_price

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
memory = MemorySaver()

system_message = "Você é um agente analista financeiro e deve utilizar suas ferramentas para responder o usuário."

tools = [
    get_company_info, 
    get_current_stock_price, 
    get_history_stock_price
    ]

agent_executor = create_react_agent(
    model=llm,
    tools=tools,
    prompt=system_message,
    checkpointer=memory
)

config = {"configurable": {"thread_id": "1"}}

while True:
    input_message = {
        "role": "user",
        "content": input("Digite: "),
    }
    for step in agent_executor.stream(
        {"messages": [input_message]}, config, stream_mode = "values"
    ):
        step["messages"][-1].pretty_print()
