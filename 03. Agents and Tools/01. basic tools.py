from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

# This code demonstrates the use of DuckDuckGo search tool for querying the web
ddg_search = DuckDuckGoSearchRun()
result = ddg_search.run("Quem foi Alan Turing?")
# print(result)

# This code demonstrates the use python REPL for executing Python code
python_repl = PythonREPL()
result = python_repl.run("print(2 + 2)")
# print(result)

# This code demonstrates the use of Wikipedia API wrapper for querying Wikipedia
wikipedia = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        lang="pt",  # Specify the language for Wikipedia queries
        extract_format="html",  # Specify the format for the extracted content
    )
)
result = wikipedia.run("Alan Turing")
# print(result)

