import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

load_dotenv()


st.set_page_config(
    page_title="Estoque", 
    page_icon=":robot_face:"
)
st.header("Assistente de Estoque")

model_options = [
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4o-mini",
    "gpt-4o",
    "gemini-1.5-flash-latest",
]

with st.sidebar:
    selected_model = st.sidebar.selectbox(
        label="Selecione o modelo LLM",
        options=model_options,
        index=model_options.index("gemini-1.5-flash-latest"),
        help="Selecione o modelo LLM que deseja usar para responder às perguntas sobre o estoque"
    )
    st.sidebar.markdown("""
    ## Sobre o modelo
    Esse agente apenas de estudo e foi implementado apenas conexão com o modelo LLM da Google
    """)

api_key_google = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, api_key=api_key_google)
# llm = ChatGoogleGenerativeAI(model=selected_model, temperature=0, api_key=api_key_google)

db = SQLDatabase.from_uri("sqlite:///db/estoque.db")
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

system_message = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    tools=toolkit.get_tools(),
    prompt=system_message,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=toolkit.get_tools(),
)

prompt = """"
Use as ferramentas necessárias para responder perguntas relacionadas ao
estoque de produtos. Você forncerá insights sobre produtos, preços, reposições de estoque
e relatórios confrome solicitado pelo usuário.
A resposta final deve ter formatação amigável de visualização para o usuário.
Sempre responda em português brasileiro.
Pergunta: {q}
"""

prompt_template = PromptTemplate.from_template(prompt)

st.write("Faça uma pergunta sobre o estoque de produtos, preços e reposições.")
question = st.text_input("O que desaja saber sobre o estoque?",)
if st.button("Consultar"):
    if question:
        with st.spinner("Consultando o modelo..."):
            formatted_question = prompt_template.format(q=question)
            output = agent_executor.invoke({"input": formatted_question})
            st.write("Resposta do modelo:")
            st.markdown(output.get("output", "Nenhuma resposta encontrada."))
    else:
        st.error("Por favor, digite uma pergunta antes de enviar.")