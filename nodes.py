from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

from state import ResearchState
from tools import search_notes, search_web
import os

load_dotenv()

llm = ChatGroq(
    model_name = os.environ['GROQ_MODEL']
)

embeddings = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
)

vector_store = Chroma(
    persist_directory = './chroma_db/',
    embedding_function = embeddings,
    collection_name = 'notes'
)

#Function used for semantic retrieval before the router node makes a routing decision
def check_notes_relevance(question, threshold=0.8):
    """
    Performs fast semantic retrieval to check if notes are relevant to the question.
    Returns True if the best matching note exceeds the relevance threshold.
    
    Args:
        question: The user's question to check against notes
        threshold: Similarity score threshold (0-1). Default 0.5.
    
    Returns:
        Boolean indicating if relevant notes were found
    """
    # Retrieve only 1 most similar document for speed
    results = vector_store.similarity_search_with_score(question, k=1)
    
    # If no documents found, notes are not relevant
    if not results:
        return False
    
    # Extract the similarity score (ignore the document itself)
    _, similarity_score = results[0]   
    
    # Return whether the similarity score meets the threshold
    return similarity_score < threshold


#Prompt for llm call in Query planner node
query_planner_prompt = ChatPromptTemplate.from_template(
    """Analyze the user's question and break it down strategically.

    Question: {query}

    Instructions:
    - If the question is simple and focused, return it as-is (a single line).
    - If the question is complex or multi-faceted, break it into 1-3 focused sub-questions that address distinct aspects.
    - Ensure sub-questions are specific and can be researched independently.
    - Preserve the core intent of the original question.
    
    Return ONLY the question(s), one per line, no numbering or extra text."""
)


#Prompt for llm call in routing node
router_prompt = ChatPromptTemplate.from_template(
    """For this question, determine if you need current web info, personal notes, or both.

    IMPORTANT: {notes_availability}
    If no relevant notes are found, you CANNOT choose "notes" or "both" - you must choose "web".
    
    Question: {query}

    Answer with one word only: web, notes, or both."""
    )

#Prompt for llm call in synthesizer node
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

def query_planner_node(state: ResearchState) -> dict:
    
    question = state['question']
    
    response = llm.invoke(
        query_planner_prompt.format_messages(query=question)
    )
    
    # Parse response.content: split by newlines and clean up each line
    sub_questions = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
    
    return {"sub_questions": sub_questions, "current_index": 0, "sub_results": []}

def route_node(state: ResearchState) -> dict:
    
    query = state['question']
    
    # Check if notes are semantically relevant to the question
    has_relevant_notes = check_notes_relevance(query)
    notes_availability = "You have relevant notes available" if has_relevant_notes else "No relevant notes found"
    
    response = llm.invoke(
        router_prompt.format_messages(query=query, notes_availability=notes_availability)
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


if __name__ == "__main__":
    # # Test questions
    # test_questions = [
    #     "What is machine learning?",
    #     "What is gradient descent?",
    #     "What is the 5th amendment in US Law?",
    #     "What is Python?",
    #     "What is the latest iPhone?",
    #     "What is the 5th Amendment in US Law"
    # ]
    
    # print("Testing semantic retrieval...\n")
    # for question in test_questions:
    #     has_relevant = check_notes_relevance(question, threshold=0.5)
    #     print(f"Q: {question}")
    #     print(f"Relevant notes found: {has_relevant}\n")
    
    # query_planner_node("What is Machine Learning and what is the latest model released by OPENAI for CHATGPT?")
    pass