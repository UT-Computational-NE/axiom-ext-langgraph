# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""The drawable contract, read off a graph LangGraph actually compiled.

Unit tests drive the same functions with a hand-built drawable. This one
checks that ``StateGraph.compile().get_graph()`` is that drawable.
"""

from __future__ import annotations

from typing import TypedDict

import pytest

pytest.importorskip("langgraph", reason="langgraph is an optional extra")

from langgraph.constants import END
from langgraph.constants import START as LG_START
from langgraph.graph import START, StateGraph

from axiom_ext_langgraph.graphwork import END as OUR_END
from axiom_ext_langgraph.graphwork import START as OUR_START
from axiom_ext_langgraph.graphwork import (
    dry_run,
    inspect_graph,
    shape_from_compiled,
    to_mermaid,
    validate_graph,
)

pytestmark = pytest.mark.graph


class _State(TypedDict):
    n: int


def _bump(state: _State) -> _State:
    return {"n": state["n"] + 1}


def _keep(state: _State) -> _State:
    return state


def _route(state: _State) -> str:
    return "again" if state["n"] < 3 else "stop"


def _compiled():
    graph = StateGraph(_State)
    graph.add_node("a", _bump)
    graph.add_node("b", _keep)
    graph.add_node("orphan", _bump)
    graph.add_edge(START, "a")
    graph.add_conditional_edges("a", _route, {"again": "b", "stop": END})
    graph.add_edge("b", "a")
    return graph.compile()


def test_compiled_sentinels_match_langgraph():
    assert LG_START == OUR_START == "__start__"
    assert END == OUR_END == "__end__"


def test_a_compiled_graph_reports_the_orphan_the_cycle_and_the_exit():
    shape = shape_from_compiled(_compiled())
    report = validate_graph(shape)

    assert report["nodes"] == ["a", "b", "orphan"]
    assert report["entries"] == ["a"]
    assert report["exits"] == ["a"]
    assert report["unreachable"] == ["orphan"]
    assert report["cycles"] == [["a", "b", "a"]]
    assert report["ok"] is False
    assert {edge["label"] for edge in report["edges"]} >= {"again", "stop"}
    assert "again" in to_mermaid(shape)

    traced = dry_run(shape, choices={"a": "stop"})
    assert [step["node"] for step in traced["traces"][0]["steps"]] == ["a", "__end__"]
    assert inspect_graph(shape)["nodes"] == report["nodes"]
