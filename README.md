# Research Assistant

A simple agentic research assistant built with **LangGraph** that can research a user's question using multiple specialized agents and combine their findings into a final answer.

## Architecture

```text
                    Question
                       │
                       ▼
                ┌──────────────┐
                │ Router Node  │
                └──────┬───────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
       ┌────────────┐    ┌────────────┐
       │  Web Agent │    │  Research  │
       │    Node    │    │    Node    │
       └──────┬─────┘    └──────┬─────┘
              │                 │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │  Synthesizer    │
              │      Node       │
              └────────┬────────┘
                       ▼
                 Final Answer
```

## Goal

The main goal of this project is to learn the fundamentals of **Agentic AI and LangGraph** by building a small multi-node research workflow.

The project will explore:

* Graph-based agent orchestration
* State management
* Conditional routing
* Parallel node execution
* Tool calling
* Web research
* Information synthesis

## How It Works

1. The user submits a research question.
2. The **Router Node** determines which research tasks are required.
3. The selected research nodes execute, potentially in parallel.
4. The **Synthesizer Node** combines the collected information.
5. The system returns a final answer to the user.

## Tech Stack

* Python
* LangGraph
* LangChain
* LLM API
* Web search API
* Vector database *(optional in later versions)*

## Future Improvements

The initial workflow can be extended with additional nodes such as:

* Query Planner
* Source Validator
* Fact Checker
* Critic
* Answer Reviser
* Citation Formatter
* Memory
* Human-in-the-loop approval

The project will be developed incrementally so that each addition demonstrates a new LangGraph concept.
