from __future__ import annotations

import random
import string
import time

import agent
import pytest

# the real solution module


@pytest.fixture(scope="function")
def app(mocker):
    get_llm = mocker.patch("agent.get_llm")
    get_llm.return_value = None

    import agent

    return agent.compile_research_graph()


def oracle(query: str) -> tuple[list[dict], str]:
    results = agent.mock_search(query)
    summary = agent._mock_summary(query, results)
    return results, summary


def test_small_exhaustive(app) -> None:
    queries = [
        "fintech regulation in Hong Kong",
        "AI agents in banking",
        "cross-border payments",
        "",  # empty is a valid edge input, checked again below
    ]

    for q in queries:
        out = app.invoke({"query": q})
        expected_results, expected_summary = oracle(q)
        assert out["results"] == expected_results, f"results mismatch for {q!r}"
        assert out["summary"] == expected_summary, f"summary mismatch for {q!r}"

    print(f"(a) small exhaustive: {len(queries)} queries OK")


def test_random_medium(app, n: int = 200):
    for _ in range(n):
        q = "".join(random.choices(string.ascii_letters + " ", k=random.randint(1, 40)))
        out = app.invoke({"query": q})
        assert isinstance(out["results"], list) and len(out["results"]) == 3
        for r in out["results"]:
            assert {"title", "snippet", "url", "date", "relevance"} <= r.keys()
        assert isinstance(out["summary"], str) and out["summary"]
        assert out["results"] == oracle(q)[0]
    print(f"(b) random medium: {n} random queries OK")


def test_scaling(app) -> None:
    print("(c) scaling / timing:")
    prev_per = None
    for n in (100, 1000, 10000):
        t0 = time.perf_counter()
        for _ in range(n):
            app.invoke({"query": "scaling probe"})
        dt = time.perf_counter() - t0
        per = dt / n
        print(f"    n={n:>6}  total={dt:7.3f}s  per_invoke={per*1e6:8.1f}us")
        if prev_per is not None:
            # per-invoke cost must stay roughly flat (allow slack); never explode
            assert per < prev_per * 5, "per-invoke time grew super-linearly"
        prev_per = per


def test_edge_cases(app) -> None:
    for q in ["", "x" * 5000, "emoji 🚀 unicode ✓ 漢字", "a\nb\tc"]:
        out = app.invoke({"query": q})
        assert out["results"] == oracle(q)[0]
        assert isinstance(out["summary"], str) and out["summary"]
    print("(d) edge cases: OK")
