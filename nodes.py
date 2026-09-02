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
        """You are a research synthesizer. Your task is to integrate and frame the research findings into a comprehensive, well-structured answer.

        Original question: {original_question}
        
        Research findings from sub-questions:
        {context}
        
        Instructions:
        - Synthesize the findings into a cohesive narrative that directly addresses the original question
        - For every distinct sub-question, explicitly identify the source used immediately after addressing it using exactly one of: "Source: Web", "Source: Notes", or "Source: Web + Notes"
        - Use the provided source for each sub-question; do not infer or change it from the research content
        - Organize information logically with clear connections between concepts
        - Frame findings in context of the original question - explain how each part contributes to answering it
        - Eliminate redundancy while preserving important insights
        - Create smooth transitions between different aspects of the answer
        - Start with the most relevant/important insights
        - Ensure the final answer reads as a unified response, not a collection of separate answers
        
        Provide a comprehensive, well-structured answer that integrates all the research findings."""
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
    
    query = state['sub_questions'][state['current_index']]
    
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
    
    query = state['sub_questions'][state['current_index']]
    answer = search_web.invoke({"query": query})
    
    return {"web_results": answer}


def notes_node(state: ResearchState) -> dict:
    
    query = state['sub_questions'][state['current_index']]
    answer = search_notes.invoke({"query": query})
    
    return {"notes_results": answer}


def process_result_node(state: ResearchState) -> dict:
    """Store current sub-question result and decide if we loop"""
    current_sub_question = state["sub_questions"][state["current_index"]]
    
    # Combine web + notes results for this sub-question
    result = {
        "question": current_sub_question,
        "web_results": state["web_results"],
        "notes_results": state["notes_results"],
        "route": state["route"]
    }
    
    # Append to sub_results
    updated_sub_results = state["sub_results"] + [result]
    
    # Increment index for next sub-question
    next_index = state["current_index"] + 1
    
    return {
        "sub_results": updated_sub_results,
        "current_index": next_index,
        "web_results": "",    # reset for next iteration
        "notes_results": ""
    }


def synthesizer_node(state: ResearchState) -> dict:
    # print("\n=== SYNTHESIZER DEBUG ===")
    # print(f"Sub-results count: {len(state['sub_results'])}")
    # for i, result in enumerate(state["sub_results"]):
    #     print(f"\nSub-Q {i+1}: {result['question']}")
    #     print(f"  Route: {result['route']}")
    #     print(f"  Web: {result['web_results'][:100]}..." if result['web_results'] else "  Web: (empty)")
    #     print(f"  Notes: {result['notes_results'][:100]}..." if result['notes_results'] else "  Notes: (empty)")
    # print("======================\n")
    
    query = state["question"]
    
     # Combine all sub-results
    combined_context = ""
    for result in state["sub_results"]:
        source_label = {"web": "Web", "notes": "Notes", "both": "Web + Notes"}[result["route"]]
        combined_context += f"Q: {result['question']}\n"
        combined_context += f"Source used: {source_label}\n"
        combined_context += f"Web: {result['web_results']}\n"
        combined_context += f"Notes: {result['notes_results']}\n\n"
    
    
    response = llm.invoke(
        synth_prompt.format_messages(
            context=combined_context,
            original_question=state["question"]
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