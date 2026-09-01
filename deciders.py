from state import ResearchState


def decide_route(state: ResearchState) -> str:
    return state["route"]  # returns "web", "notes", or "both"


def is_both(state: ResearchState) -> str:
    return "both" if state["route"] == "both" else "single"


def should_continue_loop(state: ResearchState) -> str:
    """Check if there are more sub-questions"""
    if state["current_index"] < len(state["sub_questions"]):
        return "continue"  # loop back to router
    else:
        return "synthesize"  # done with all sub-questions
