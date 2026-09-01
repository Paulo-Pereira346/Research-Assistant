from typing import TypedDict

class ResearchState(TypedDict):
    question: str
    sub_questions: list[str]
    current_index: int
    sub_results: list[dict]   
    web_results: str
    notes_results: str
    final_answer: str
    route: str