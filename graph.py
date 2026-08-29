from langgraph.graph import StateGraph, END
from state import ResearchState
from nodes import route_node, web_node, notes_node, synthesizer_node

graph = StateGraph(ResearchState)

graph.add_node("router", route_node)
graph.add_node("web", web_node)
graph.add_node("notes", notes_node)
graph.add_node("synthesizer", synthesizer_node)

graph.set_entry_point("router")

def decide_route(state: ResearchState) -> str:
    return state["route"]  # returns "web", "notes", or "both"

graph.add_conditional_edges(
    "router",
    decide_route,
    {
        "web": "web",
        "notes": "notes",
        "both": "web"
    }
)

def is_both(state: ResearchState) -> str:
    return "both" if state["route"] == "both" else "single"

graph.add_conditional_edges(
    "web",
    is_both,
    {
        "both": "notes",
        "single": "synthesizer"
    }
)

graph.add_edge("notes", "synthesizer")
graph.add_edge("synthesizer", END)

app = graph.compile()

def run(question: str):
    result = app.invoke({
        "question": question,
        "web_results": "",
        "notes_results": "",
        "final_answer": "",
        "route": ""
    })
    return result["final_answer"], result["route"]

if __name__ == "__main__":
    # Test the graph
    test_question = "What is machine learning and what is the latest AI model released by OpenAI?"
    answer, route = run(test_question)
    
    print(f"Question: {test_question}")
    print(f"Answer: {answer}")
    print(f"Choice: {route}")