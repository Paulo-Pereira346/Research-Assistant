from langgraph.graph import StateGraph, END
from state import ResearchState
from nodes import query_planner_node, route_node, web_node, notes_node, process_result_node, synthesizer_node
from deciders import decide_route, is_both, should_continue_loop

graph = StateGraph(ResearchState)

graph.add_node("query_planner", query_planner_node)
graph.add_node("router", route_node)
graph.add_node("web", web_node)
graph.add_node("notes", notes_node)
graph.add_node("process_result", process_result_node)
graph.add_node("synthesizer", synthesizer_node)

graph.set_entry_point("query_planner")

graph.add_edge("query_planner", "router")

graph.add_conditional_edges(
    "router",
    decide_route,
    {
        "web": "web",
        "notes": "notes",
        "both": "web"
    }
)

graph.add_conditional_edges(
    "web",
    is_both,
    {
        "both": "notes",
        "single": "process_result"
    }
)


graph.add_conditional_edges(
    "process_result", 
    should_continue_loop, 
    {
        "continue": "router",
        "synthesize": "synthesizer"
    }
)

graph.add_edge("notes", "process_result")
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

    routes = {sub_result["route"] for sub_result in result["sub_results"]}
    overall_route = "both" if len(routes) > 1 or "both" in routes else routes.pop()

    return result["final_answer"], overall_route

if __name__ == "__main__":
    # Test the graph
    test_question = "Compare supervised learning with the latest reinforcement learning techniques. Also explain what neural networks are and how they differ from classical algorithms."
    answer, route = run(test_question)
    
    print(f"Question: {test_question}")
    print(f"Answer: {answer}")
    print(f"Choice: {route}")