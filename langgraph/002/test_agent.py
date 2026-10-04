"""Day 02 harness — imports the real agent.py, forces the mock classifier,
checks category + confidence + ROUTING against an oracle, and times scaling."""
from __future__ import annotations

import random
import time

import pytest

import agent


@pytest.fixture(scope="function")
def app(mocker):
    mocker.patch("agent.get_llm").return_value = None  # force deterministic mock path
    return agent.compile_classifier_graph(threshold=0.8)


THRESHOLD = 0.8


def oracle(subject: str, body: str, threshold: float = THRESHOLD) -> dict:
    """Reference answer computed WITHOUT the graph: classify + route."""
    cat, conf = agent.mock_classify(subject, body)
    action = "auto_response" if conf >= threshold else "human_review"
    return {"category": cat, "confidence": conf, "action": action}


def test_small_exhaustive(app) -> None:
    tickets = [
        ("Cannot access my account", "locked out after 3 failed login attempts"),   # account, high
        ("Refund my invoice", "I was billed twice for my subscription"),            # billing, high
        ("Unauthorized charge", "suspicious stolen card activity, possible fraud"), # fraud, high
        ("App crashes", "the bug makes it crash, totally broken"),                  # technical, high
        ("Weird thing happened", "I'm not sure what this is about"),                # general, low
        ("", ""),                                                                   # empty -> low
    ]
    for subj, body in tickets:
        out = app.invoke({"ticket_subject": subj, "ticket_body": body})
        exp = oracle(subj, body)
        assert out["category"] == exp["category"], f"cat {subj!r}"
        assert out["confidence"] == exp["confidence"], f"conf {subj!r}"
        assert out["action"] == exp["action"], f"route {subj!r}"
        assert out["response"]  # a response string was produced on both branches
    print(f"(a) small exhaustive: {len(tickets)} tickets OK")


def test_routing_boundary() -> None:
    """Router is pure: test it directly around the >= 0.8 edge."""
    route = agent.make_router(0.8)
    assert route({"confidence": 0.80}) == "auto"    # boundary is inclusive (>=)
    assert route({"confidence": 0.799}) == "human"
    assert route({"confidence": 1.0}) == "auto"
    assert route({"confidence": 0.0}) == "human"
    print("(b) routing boundary: OK")


def test_random_medium(app, n: int = 200) -> None:
    corpus = {
        "billing": "refund invoice payment billed",
        "fraud": "fraud unauthorized scam stolen",
        "account": "login locked password access",
        "technical": "error bug crash broken",
        "general": "hello just saying hi",
    }
    cats = list(corpus)
    for _ in range(n):
        c = random.choice(cats)
        body = corpus[c]
        out = app.invoke({"ticket_subject": c, "ticket_body": body})
        exp = oracle(c, body)
        assert out["category"] == exp["category"]
        assert out["action"] == exp["action"]
        assert out["action"] in ("auto_response", "human_review")
    print(f"(c) random medium: {n} tickets OK")


def test_scaling(app) -> None:
    print("(d) scaling / timing:")
    prev = None
    for n in (100, 1000, 10000):
        t0 = time.perf_counter()
        for _ in range(n):
            app.invoke({"ticket_subject": "login locked", "ticket_body": "password access"})
        per = (time.perf_counter() - t0) / n
        print(f"    n={n:>6}  per_invoke={per*1e6:8.1f}us")
        if prev is not None:
            assert per < prev * 5, "per-invoke time grew super-linearly"
        prev = per


def test_edge_cases(app) -> None:
    for subj, body in [("", ""), ("x" * 5000, "y" * 5000), ("漢字 🚀", "emoji ✓"), ("a\nb", "c\td")]:
        out = app.invoke({"ticket_subject": subj, "ticket_body": body})
        exp = oracle(subj, body)
        assert out["category"] == exp["category"]
        assert out["action"] == exp["action"]
        assert out["response"]
    print("(e) edge cases: OK")
