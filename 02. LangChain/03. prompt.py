import os
from getpass import getpass
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# --- 1. Configuração do Ambiente ---
# Garanta que sua GOOGLE_API_KEY esteja configurada.
# Se não estiver nas variáveis de ambiente, ela será solicitada.
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass("Insira sua chave de API do Google AI: ")

# --- 2. Inicializando o Modelo de Linguagem (LLM) ---
# Usamos 'gemini-pro' como o modelo principal.
# 'temperature=0' para respostas mais diretas e consistentes na tradução.
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

# --- 3. Definindo o PromptTemplate para Tradução ---
# Este template permite criar prompts de tradução flexíveis,
# substituindo os placeholders {idioma1}, {idioma2} e {texto}.
template = """
Traduza o texto do {idioma1} para o {idioma2}:
{texto}
"""

# Cria uma instância de PromptTemplate a partir do template de string.
prompt_template = PromptTemplate.from_template(template=template)

# --- 4. Formatando o Prompt ---
# Aqui, preenchemos os placeholders do template com valores específicos para a tradução.
# Exemplo 1: Português para Inglês
idioma_origem_1 = "português"
idioma_destino_1 = "inglês"
texto_original_1 = "Olá, como você está?"

prompt_formatado_1 = prompt_template.format(
    idioma1=idioma_origem_1,
    idioma2=idioma_destino_1,
    texto=texto_original_1
)

# Exemplo 2: Inglês para Espanhol
idioma_origem_2 = "inglês"
idioma_destino_2 = "espanhol"
texto_original_2 = "The quick brown fox jumps over the lazy dog."

prompt_formatado_2 = prompt_template.format(
    idioma1=idioma_origem_2,
    idioma2=idioma_destino_2,
    texto=texto_original_2
)

# --- 5. Invocando o LLM com os Prompts Formatados ---

print("--- Tradução 1: Português para Inglês ---")
print(f"Texto Original ({idioma_origem_1}): '{texto_original_1}'")
try:
    # O LLM espera uma lista de objetos de mensagem.
    # Colocamos o prompt formatado dentro de um HumanMessage.
    response_1 = llm.invoke([HumanMessage(content=prompt_formatado_1)])
    print(f"Texto Traduzido ({idioma_destino_1}): '{response_1.content}'")
except Exception as e:
    print(f"Ocorreu um erro na tradução 1: {e}")
    print("Verifique sua GOOGLE_API_KEY e acesso ao modelo 'gemini-pro'.")

print("\n--- Tradução 2: Inglês para Espanhol ---")
print(f"Texto Original ({idioma_origem_2}): '{texto_original_2}'")
try:
    response_2 = llm.invoke([HumanMessage(content=prompt_formatado_2)])
    print(f"Texto Traduzido ({idioma_destino_2}): '{response_2.content}'")
except Exception as e:
    print(f"Ocorreu um erro na tradução 2: {e}")
    print("Verifique sua GOOGLE_API_KEY e acesso ao modelo 'gemini-pro'.")

# --- 6. Exemplo com uma Mensagem de Sistema Adicional (Opcional) ---
print("\n--- Tradução 3: Com Mensagem de Sistema ---")
idioma_origem_3 = "francês"
idioma_destino_3 = "português"
texto_original_3 = "Bonjour, comment allez-vous?"

# O prompt formatado ainda é o mesmo, mas a mensagem de sistema dá mais contexto.
prompt_formatado_3 = prompt_template.format(
    idioma1=idioma_origem_3,
    idioma2=idioma_destino_3,
    texto=texto_original_3
)

messages_with_system_prompt = [
    SystemMessage(content="Você é um tradutor profissional e preciso. Sua única tarefa é traduzir textos. Não adicione comentários."),
    HumanMessage(content=prompt_formatado_3)
]

print(f"Texto Original ({idioma_origem_3}): '{texto_original_3}'")
try:
    response_3 = llm.invoke(messages_with_system_prompt)
    print(f"Texto Traduzido ({idioma_destino_3}): '{response_3.content}'")
except Exception as e:
    print(f"Ocorreu um erro na tradução 3: {e}")
    print("Verifique sua GOOGLE_API_KEY e acesso ao modelo 'gemini-pro'.")