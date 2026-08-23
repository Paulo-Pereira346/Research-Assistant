from typing import TypedDict

class ResearchState(TypedDict):
    question: str
    web_results: str
    notes_results: str
    final_answer: str
    route: str