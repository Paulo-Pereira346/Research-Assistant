from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from state import ResearchState
from tools import search_notes, search_web
import os

load_dotenv()

llm = ChatGroq(
    model_name = os.environ['GROQ_MODEL']
)

router_prompt = ChatPromptTemplate.from_template(
    """For this question, do I need current web info, can I answer from personal notes,
    or do I need both?
    Question: {query}

    Answer with one word only: web, notes, or both."""
    )

synth_prompt = ChatPromptTemplate.from_template(
                """
                Answer the user's question using the available research below.

                Routing decision: {route}
                User question: {query}

                Web research:
                {web}

                Personal notes:
                {notes}

                Instructions:
                - If the routing decision is "web", use the web research as the source.
                - If the routing decision is "notes", use the personal notes as the source.
                - If the routing decision is "both", combine both sources and reconcile
                    conflicting information carefully.
                - Ignore a source if it is empty or unavailable, even when it is shown above.
                - Do not claim information came from a source that does not support it.
                - Give a clear, direct answer to the user's question.
        """
    )

def route_node(state: ResearchState) -> dict:
    
    query = state['question']

    
    response = llm.invoke(
    router_prompt.format_messages(query=query)
    )

    route = response.content.strip().lower()
    
    if route not in ["web", "notes", "both"]:
        route = "both"  # safe default
    
    return {"route": route}

    
def web_node(state: ResearchState) -> dict:
    
    query = state['question']
    answer = search_web.invoke({"query": query})
    
    return {"web_results": answer}


def notes_node(state: ResearchState) -> dict:
    
    query = state['question']
    answer = search_notes.invoke({"query": query})
    
    return {"notes_results": answer}


def synthesizer_node(state: ResearchState) -> dict:
    query = state["question"]
    web_results = state.get("web_results", "")
    notes_results = state.get("notes_results", "")
    route = state["route"]
    
    
    response = llm.invoke(
        synth_prompt.format_messages(
            query = query,
            web = web_results,
            notes = notes_results,
            route = route
        )
    )

    return {"final_answer": response.content}
