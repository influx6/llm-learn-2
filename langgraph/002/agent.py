from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Literal, TypedDict, get_args

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from pytest import Class


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


CategoryTypes = Literal["billing", "technical", "fraud", "account", "general"]

CATEGORIES = get_args(CategoryTypes)

# ---- Deterministic keyword classifier (mock path, used by tests). ----
KEYWORDS = {
    "billing": ["invoice", "charge", "refund", "payment", "billed", "subscription"],
    "fraud": ["fraud", "unauthorized", "scam", "stolen", "suspicious"],
    "account": ["login", "locked", "password", "access", "account", "sign in"],
    "technical": ["error", "bug", "crash", "not working", "broken", "fails"],
}


class State(TypedDict):
    ticket_subject: str
    ticket_body: str
    category: str
    confidence: float
    action: str
    response: str
    agent: str


# -- Structured schema for the reall LLM path: a typed, validated answer.
class Classification(BaseModel):
    category: CategoryTypes
    confidence: float = Field(ge=0.0, le=1.0)
    agent: str


def mock_classify(subject: str, body: str) -> tuple[str, float]:
    text = f"{subject} {body}".lower()
    best, hits = "general", 0
    for cat, words in KEYWORDS.items():
        c = sum(w in text for w in words)
        if c > hits:
            best, hits = cat, c
    if hits == 0:
        return "general", 0.3  # nothing matched -> deliberately low confidence
    return best, min(0.6 + (0.2 * hits), 0.99)


def classify_node(state: State) -> dict:
    llm = get_llm()
    if llm is None:
        cat, conf = mock_classify(state["ticket_subject"], state["ticket_body"])
        return {"category": cat, "confidence": conf, "agent": "mock"}
    structured = llm.with_structured_output(Classification)
    prompt = (
        f"Classify this support ticket into exactly one of {CATEGORIES}.\n"
        f"Subject: {state['ticket_subject']}\nBody: {state['ticket_body']}\n"
        "Return the category and your confidence between 0 and 1."
        "Ensure to include a 'agent' field in the dict with "
        "your name (what model you are, not just role name) so we know who generate the result."
    )
    result: Classification = structured.invoke(prompt)  # type: ignore
    assert isinstance(result, Classification), "must be a instance of Classification"

    return result.model_dump(mode="python", by_alias=True)


def auto_response_node(state: State) -> dict:
    return {
        "action": "auto_response",
        "response": f"Thanks - your {state['category']} issue is being handled automatically.",
    }


def human_review_node(state: State) -> dict:
    return {
        "action": "human_review",
        "response": (
            f"Flagged for human review "
            f"(category={state['category']}, confidence={state['confidence']:.2f})"
        ),
    }


# --- Router: pure, read-only, returns a LABEL (not a node, not an update). ---
def make_router(threshold: float):
    def route_by_confidence(state: State) -> str:
        return "auto" if state["confidence"] >= threshold else "human"

    return route_by_confidence


def compile_research_graph(threshold: float = 0.8):
    g = StateGraph(State)

    # g.add_node("search", search_node)
    # g.add_edge(START, "search")
    # g.add_edge("summarize", END)

    g.add_node("classify", classify_node)
    g.add_node("auto_response", auto_response_node)
    g.add_node("human_review", human_review_node)

    g.add_edge(START, "classify")
    g.add_conditional_edges(
        "classify",
        make_router(threshold),
        {
            "auto": "auto_response",
            "human": "human_review",
        },
    )
    g.add_edge("auto_response", END)
    g.add_edge("human_review", END)

    return g.compile()


if __name__ == "__main__":
    app = compile_research_graph()
    hi = app.invoke(
        {
            "ticket_subject": "Cannot access my account",
            "ticket_body": "I've been locked out after 3 failed login attempts",
        }
    )

    print("HIGH: ", hi["category"], hi["confidence"], hi["action"], hi)

    lo = app.invoke(
        {
            "ticket_subject": "Weird thing happened",
            "ticket_body": "I'm not sure what this is about",
        }
    )

    print("LOW: ", lo["category"], lo["confidence"], lo["action"], lo)
