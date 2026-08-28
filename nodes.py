from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from state import ResearchState
from tools import search_notes, search_web
import os

load_dotenv()

llm = ChatGroq(
    model_name = os.environ['GROQ_MODEL']
)

def route_node(state: ResearchState) -> dict:
    
    query = state['question']
    
    prompt = ChatPromptTemplate.from_template(
    """For this question, do I need current web info, can I answer from personal notes,
    or do I need both?
    Question: {query}

    Answer with one word only: web, notes, or both."""
    )
    
    response = llm.invoke(
    prompt.format_messages(query=query)
    )

    route = response.content.strip().lower()
    
    return {"route": route}
    
def web_node(state: ResearchState) -> dict:
    
    query = state['question']
    answer = search_web.invoke({"query": query})
    
    return {"web_results": answer}

def notes_node(state: ResearchState) -> dict:
    
    query = state['question']
    answer = search_notes.invoke({"query": query})
    
    return {"notes_results": answer}

if __name__ == "__main__":
    print(route_node("What is the best gpt model and what is the latest one?"))
