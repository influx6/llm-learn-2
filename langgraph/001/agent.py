from __future__ import annotations

import os
from collections.abc import Iterable
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


def get_first(items: Iterable[tuple], checker=lambda x: x is not None):
    for item in items:
        if checker(item):
            return item
    return (None, None, None)


def get_llm():
    api_key, api_endpoint, api_model = get_first(
        [
            (
                os.environ.get("OPENROUTER_API_KEY", None),
                "https://openrouter.ai/api/v1",
                "deepseek/deepseek-chat",
            ),
            (
                os.environ.get("DEEPSEEK_AI_KEY", None),
                "https://api.deepseek.com",
                "deepseek-flash",
            ),
        ],
        checker=lambda x: x[0] is not None,
    )
    if not api_key:
        return None

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        base_url=api_endpoint,
        api_key=api_key,
        model=api_model,
        temperature=0.2,
    )


# the state we share across to every nodes (which are functions that take this state)
# basically we tell the graph this is the state you will use, it handles updating it
# as each node provides output, hence the typed dict.
class State(TypedDict):
    query: str
    summary: str
    results: list[dict]


def mock_search(query: str) -> list[dict]:
    return [
        {
            "title": f"{query}: regulators tighten disclosure rules",
            "snippet": "New guidance expands reporting obligations for licensed firms.",
            "url": "https://example.com/a",
            "date": "2026-01-10",
            "relevance": 0.94,
        },
        {
            "title": f"{query}: sandbox expands to more startups",
            "snippet": "The regulator widened its fintech sandbox to onboard more participants.",
            "url": "https://example.com/b",
            "date": "2026-02-02",
            "relevance": 0.88,
        },
        {
            "title": f"{query}: cross-border payment pilot results",
            "snippet": "A pilot cut settlement times and flagged new compliance checkpoints.",
            "url": "https://example.com/c",
            "date": "2026-02-20",
            "relevance": 0.81,
        },
    ]


def _mock_summary(query: str, results: list[dict]) -> str:
    bullets = "; ".join(r["title"] for r in results)
    return f"Summary for '{query}' (mock): {bullets}."


# --- Node 1: search, reads the query, returns only the key it changed.
def search_node(state: State) -> dict:
    return {"results": mock_search(state["query"])}


# -- Node 2: summarize. Reads results, calls the LLM, returns summary.
def summarize_node(state: State) -> dict:
    llm = get_llm()
    if not llm:
        return {"summary": _mock_summary(state["query"], state["results"])}

    query = state["query"]
    results = state["results"]
    findings = "\n".join(f"- {r['title']}: {r['snippet']}" for r in results)
    prompt = (
        f"Summarize the following search findings about "
        f"'{query}' in 3-4 sentences for a busy analyst: \n\n{findings}"
    )

    response = llm.invoke(prompt)
    return {"summary": response.content}


def compile_research_graph():
    g = StateGraph(State)
    g.add_node("search", search_node)
    g.add_node("summarize", summarize_node)

    g.add_edge(START, "search")
    g.add_edge("search", "summarize")
    g.add_edge("summarize", END)

    return g.compile()


if __name__ == "__main__":
    app = compile_research_graph()
    out = app.invoke({"query": "latest trends in fintech regulation in Hong Kong"})
    print(out["summary"])
