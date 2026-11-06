# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""Graph inspection, validation, mermaid export, and dry-run.

The command, the tools, and the library call one implementation. These tests
construct a document; they do not need LangGraph.
"""

from __future__ import annotations

import json

import pytest

LINE = {
    "nodes": ["read", "build"],
    "edges": [
        {"source": "__start__", "target": "read"},
        {"source": "read", "target": "build"},
        {"source": "build", "target": "__end__"},
    ],
}

HEALTHY_LOOP = {
    "nodes": ["a", "b"],
    "edges": [
        {"source": "__start__", "target": "a"},
        {"source": "a", "target": "b", "conditional": True, "label": "again"},
        {"source": "a", "target": "__end__", "conditional": True, "label": "stop"},
        {"source": "b", "target": "a"},
    ],
}

WITH_ORPHAN = {
    "nodes": ["a", "b", "orphan"],
    "edges": HEALTHY_LOOP["edges"],
}

FORK = {
    "nodes": ["a", "b", "c"],
    "edges": [
        {"source": "__start__", "target": "a"},
        {"source": "a", "target": "b"},
        {"source": "a", "target": "c"},
        {"source": "b", "target": "__end__"},
        {"source": "c", "target": "__end__"},
    ],
}


def _codes(report: dict) -> set[str]:
    return {finding["code"] for finding in report["findings"]}


def test_inspect_reports_nodes_edges_entry_and_exit():
    from axiom_ext_langgraph.graphwork import inspect_graph, shape_from_document

    report = inspect_graph(shape_from_document(LINE))

    assert report["nodes"] == ["read", "build"]
    assert report["entries"] == ["read"]
    assert report["exits"] == ["build"]
    assert report["unreachable"] == []
    assert report["cycles"] == []
    assert report["edges"] == [
        {"source": "__start__", "target": "read", "conditional": False, "label": ""},
        {"source": "read", "target": "build", "conditional": False, "label": ""},
        {"source": "build", "target": "__end__", "conditional": False, "label": ""},
    ]


def test_a_healthy_loop_is_valid_and_names_its_cycle():
    from axiom_ext_langgraph.graphwork import shape_from_document, validate_graph

    report = validate_graph(shape_from_document(HEALTHY_LOOP))

    assert report["ok"] is True
    assert report["cycles"] == [["a", "b", "a"]]
    assert report["exits"] == ["a"]
    assert _codes(report) == {"cycle"}
    assert report["findings"][0]["severity"] == "cycle"


def test_an_orphan_is_unreachable_and_a_dead_end():
    from axiom_ext_langgraph.graphwork import shape_from_document, validate_graph

    report = validate_graph(shape_from_document(WITH_ORPHAN))

    assert report["ok"] is False
    assert report["unreachable"] == ["orphan"]
    assert {"unreachable", "dead_end", "cycle"} <= _codes(report)


def test_a_cycle_with_no_entry_is_not_ok():
    from axiom_ext_langgraph.graphwork import shape_from_document, validate_graph

    report = validate_graph(
        shape_from_document(
            {
                "nodes": ["a", "b"],
                "edges": [
                    {"source": "a", "target": "b"},
                    {"source": "b", "target": "a"},
                ],
            }
        )
    )

    assert report["ok"] is False
    assert "no_entry" in _codes(report)
    assert report["unreachable"] == []


def test_an_edge_to_an_unknown_node_is_an_error():
    from axiom_ext_langgraph.graphwork import shape_from_document, validate_graph

    report = validate_graph(
        shape_from_document(
            {"nodes": ["a"], "edges": [{"source": "__start__", "target": "a"}, {"source": "a", "target": "missing"}]}
        )
    )

    assert report["ok"] is False
    assert "unknown_node" in _codes(report)


def test_a_node_with_no_outgoing_edge_is_a_dead_end():
    from axiom_ext_langgraph.graphwork import shape_from_document, validate_graph

    report = validate_graph(
        shape_from_document({"nodes": ["a"], "edges": [{"source": "__start__", "target": "a"}]})
    )

    assert report["ok"] is False
    assert {"dead_end", "no_exit"} <= _codes(report)


def test_start_and_end_are_sentinels_not_nodes():
    from axiom_ext_langgraph.graphwork import END, START, shape_from_document

    assert START == "__start__"
    assert END == "__end__"
    with pytest.raises(TypeError, match="__start__"):
        shape_from_document({"nodes": ["__start__"], "edges": []})
    with pytest.raises(TypeError, match="__end__"):
        shape_from_document({"nodes": ["__end__"], "edges": []})


def test_a_document_that_is_not_the_shape_we_read_is_refused():
    from axiom_ext_langgraph.graphwork import shape_from_document

    with pytest.raises(TypeError):
        shape_from_document({"nodes": ["a"]})
    with pytest.raises(TypeError):
        shape_from_document({"nodes": ["a", "a"], "edges": []})
    with pytest.raises(TypeError):
        shape_from_document({"nodes": [""], "edges": []})
    with pytest.raises(TypeError):
        shape_from_document({"extra": True, "nodes": ["a"], "edges": []})


def test_mermaid_is_a_flowchart_of_the_same_shape():
    from axiom_ext_langgraph.graphwork import shape_from_document, to_mermaid

    assert to_mermaid(shape_from_document(LINE)) == (
        "flowchart TD\n"
        '  __start__["__start__"]\n'
        '  read["read"]\n'
        '  build["build"]\n'
        '  __end__["__end__"]\n'
        '  __start__["__start__"] --> read["read"]\n'
        '  read["read"] --> build["build"]\n'
        '  build["build"] --> __end__["__end__"]\n'
    )
    drawn = to_mermaid(shape_from_document(HEALTHY_LOOP))
    assert 'a["a"] -.->|again| b["b"]' in drawn
    assert 'a["a"] -.->|stop| __end__["__end__"]' in drawn


def test_a_node_name_that_is_not_a_mermaid_id_is_sanitized():
    from axiom_ext_langgraph.graphwork import shape_from_document, to_mermaid

    drawn = to_mermaid(
        shape_from_document(
            {
                "nodes": ["data.install"],
                "edges": [
                    {"source": "__start__", "target": "data.install"},
                    {"source": "data.install", "target": "__end__"},
                ],
            }
        )
    )

    assert 'data_install["data.install"]' in drawn


def test_dry_run_traces_a_line_without_calling_anything():
    from axiom_ext_langgraph.graphwork import dry_run, shape_from_document

    traced = dry_run(shape_from_document(LINE))

    assert traced["executed"] is False
    assert len(traced["traces"]) == 1
    trace = traced["traces"][0]
    assert trace["ended"] == "exit"
    assert [step["node"] for step in trace["steps"]] == ["read", "build", "__end__"]
    assert trace["steps"][0]["via"] == "__start__"


def test_dry_run_stops_at_an_unresolved_conditional():
    from axiom_ext_langgraph.graphwork import dry_run, shape_from_document

    traced = dry_run(shape_from_document(HEALTHY_LOOP))

    trace = traced["traces"][0]
    assert trace["ended"] == "unresolved"
    assert [step["node"] for step in trace["steps"]] == ["a"]
    assert trace["branches"] == [
        {"label": "again", "target": "b", "conditional": True},
        {"label": "stop", "target": "__end__", "conditional": True},
    ]


def test_dry_run_follows_a_named_choice_and_stops_on_the_cycle():
    from axiom_ext_langgraph.graphwork import dry_run, shape_from_document

    again = dry_run(shape_from_document(HEALTHY_LOOP), choices={"a": "again"})
    stop = dry_run(shape_from_document(HEALTHY_LOOP), choices={"a": "stop"})

    assert again["traces"][0]["ended"] == "cycle"
    assert [step["node"] for step in again["traces"][0]["steps"]] == ["a", "b", "a"]
    assert again["traces"][0]["steps"][-1]["cycle"] is True
    assert stop["traces"][0]["ended"] == "exit"
    assert [step["node"] for step in stop["traces"][0]["steps"]] == ["a", "__end__"]


def test_dry_run_forks_unconditional_parallel_edges():
    from axiom_ext_langgraph.graphwork import dry_run, shape_from_document

    traced = dry_run(shape_from_document(FORK))

    paths = [[step["node"] for step in trace["steps"]] for trace in traced["traces"]]
    assert paths == [
        ["a", "b", "__end__"],
        ["a", "c", "__end__"],
    ]
    assert {trace["ended"] for trace in traced["traces"]} == {"exit"}


def test_dry_run_honours_max_steps_and_refuses_a_bad_choice():
    from axiom_ext_langgraph.graphwork import dry_run, shape_from_document

    traced = dry_run(shape_from_document(LINE), max_steps=1)
    assert traced["traces"][0]["ended"] == "max_steps"
    assert [step["node"] for step in traced["traces"][0]["steps"]] == ["read"]

    with pytest.raises(TypeError):
        dry_run(shape_from_document(LINE), max_steps=0)
    with pytest.raises(ValueError, match="matched"):
        dry_run(shape_from_document(HEALTHY_LOOP), choices={"a": "sideways"})


def test_shape_from_compiled_reads_get_graph_and_refuses_anything_else():
    from axiom_ext_langgraph.graphwork import inspect_graph, shape_from_compiled

    class Node:
        def __init__(self, node_id: str) -> None:
            self.id = node_id

    class Edge:
        def __init__(self, source: str, target: str, conditional: bool = False, data: str | None = None):
            self.source = source
            self.target = target
            self.conditional = conditional
            self.data = data

    class Drawable:
        def __init__(self) -> None:
            self.nodes = {
                "__start__": Node("__start__"),
                "a": Node("a"),
                "__end__": Node("__end__"),
            }
            self.edges = [Edge("__start__", "a"), Edge("a", "__end__", data=None)]

    class Compiled:
        def get_graph(self) -> Drawable:
            return Drawable()

    report = inspect_graph(shape_from_compiled(Compiled()))
    assert report["nodes"] == ["a"]
    assert report["entries"] == ["a"]
    assert report["exits"] == ["a"]

    with pytest.raises(TypeError, match="get_graph"):
        shape_from_compiled(object())

    class Empty:
        def get_graph(self) -> object:
            return object()

    with pytest.raises(TypeError, match="nodes"):
        shape_from_compiled(Empty())


def test_command_tool_and_library_share_one_inspect(tmp_path, capsys):
    from axiom_ext_langgraph.commands.langgraph import cli
    from axiom_ext_langgraph.graphwork import inspect_graph, shape_from_document
    from axiom_ext_langgraph.library import inspect_graph as library_inspect
    from axiom_ext_langgraph.tools.inspect_graph import InspectGraph

    expected = inspect_graph(shape_from_document(LINE))
    assert library_inspect(shape_from_document(LINE)) == expected
    assert InspectGraph.invoke(LINE) == expected

    path = tmp_path / "graph.json"
    path.write_text(json.dumps(LINE), encoding="utf-8")
    assert cli(["inspect", str(path)]) == 0
    assert json.loads(capsys.readouterr().out) == expected


def test_validate_command_returns_nonzero_when_the_graph_is_not_ok(tmp_path, capsys):
    from axiom_ext_langgraph.commands.langgraph import cli
    from axiom_ext_langgraph.tools.validate_graph import ValidateGraph

    assert ValidateGraph.invoke(WITH_ORPHAN)["ok"] is False
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(WITH_ORPHAN), encoding="utf-8")
    assert cli(["validate", str(path)]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_validate_command_returns_zero_for_a_sound_graph(tmp_path, capsys):
    from axiom_ext_langgraph.commands.langgraph import cli

    path = tmp_path / "graph.json"
    path.write_text(json.dumps(LINE), encoding="utf-8")
    assert cli(["validate", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_mermaid_and_dry_run_commands_match_the_library(tmp_path, capsys):
    from axiom_ext_langgraph.commands.langgraph import cli
    from axiom_ext_langgraph.graphwork import dry_run, shape_from_document, to_mermaid
    from axiom_ext_langgraph.tools.dry_run_graph import DryRunGraph
    from axiom_ext_langgraph.tools.graph_mermaid import GraphMermaid

    path = tmp_path / "graph.json"
    path.write_text(json.dumps(LINE), encoding="utf-8")
    assert GraphMermaid.invoke(LINE) == {"mermaid": to_mermaid(shape_from_document(LINE))}
    assert cli(["mermaid", str(path)]) == 0
    assert capsys.readouterr().out == to_mermaid(shape_from_document(LINE))

    payload = {**HEALTHY_LOOP, "choices": {"a": "stop"}}
    assert DryRunGraph.invoke(payload) == dry_run(shape_from_document(HEALTHY_LOOP), choices={"a": "stop"})
    loop = tmp_path / "loop.json"
    loop.write_text(json.dumps(HEALTHY_LOOP), encoding="utf-8")
    assert cli(["dry-run", str(loop), "--choice", "a=stop"]) == 0
    assert json.loads(capsys.readouterr().out) == DryRunGraph.invoke(payload)


def test_a_graph_command_refuses_a_document_it_cannot_read(tmp_path, capsys):
    from axiom_ext_langgraph.commands.langgraph import cli

    path = tmp_path / "graph.json"
    path.write_text("{", encoding="utf-8")
    assert cli(["inspect", str(path)]) == 2
    captured = capsys.readouterr()
    assert captured.err
    assert captured.out == ""
