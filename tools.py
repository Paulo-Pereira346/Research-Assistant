from langchain_core.tools import tool
from tavily import TavilyClient
from dotenv import load_dotenv
from rag import ask

load_dotenv()

#Web Search Client
client = TavilyClient()

@tool
def search_notes(query: str):
    """ Search the users personal notes for answers to the question.
    Use this tool to answer questions that the user has already uploaded notes on. """
    
    answer = ask(query)
    return answer

@tool 
def search_web(query: str):
    """ Search the web for current information not available in personal notes.
    Use this for recent events, facts, or anything not in the notes. """
    
    answer = client.search(query, max_results=3)
    return str(answer)

tools = [search_notes, search_web]