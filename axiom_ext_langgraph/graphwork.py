# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""Inspect a graph without running it, and without a checkpointer.

One implementation. The ``langgraph`` command, the AEOS tools, and the library
all call these functions, so a report cannot drift between surfaces.

The shape is the drawable LangGraph already builds. ``get_graph()`` returns a
graph whose nodes have ``id`` and whose edges have ``source``, ``target``,
``conditional``, and ``data`` (the branch label, or ``None`` when the label is
the target). ``START`` and ``END`` are ``__start__`` and ``__end__``. Those
are the fields this module reads. An object that does not have them raises.

Nothing here executes a node. A dry-run walks edges. At a conditional edge it
stops, unless the caller named the branch. Cycles are reported. They do not
by themselves make a graph invalid: a tool loop is a cycle with an exit.
An unreachable node, a dead end, a missing entry, or an edge to an undeclared
node does.

This module does not install a checkpointer and does not read one.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "END",
    "START",
    "GraphDocumentError",
    "GraphEdge",
    "GraphShape",
    "dry_run",
    "inspect_graph",
    "load_graph_document",
    "shape_from_compiled",
    "shape_from_document",
    "shape_from_payload",
    "to_mermaid",
    "validate_graph",
]

START = "__start__"
END = "__end__"
_MERMAID_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class GraphDocumentError(ValueError):
    """A graph document could not be read as the shape this module inspects."""


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    conditional: bool = False
    label: str = ""


@dataclass(frozen=True)
class GraphShape:
    nodes: tuple[str, ...]
    edges: tuple[GraphEdge, ...]


def shape_from_document(document: Mapping[str, Any]) -> GraphShape:
    """Build a shape from ``{"nodes": [...], "edges": [...]}``."""
    if not isinstance(document, dict):
        raise TypeError("graph document must be an object with nodes and edges")
    unknown = sorted(set(document) - {"nodes", "edges"})
    if unknown:
        raise TypeError(f"unknown graph fields: {', '.join(unknown)}")
    raw_nodes = document.get("nodes")
    raw_edges = document.get("edges")
    if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
        raise TypeError("graph document needs nodes and edges, both lists")

    nodes: list[str] = []
    seen: set[str] = set()
    for node in raw_nodes:
        if not isinstance(node, str) or not node:
            raise TypeError("each node must be a non-empty string")
        if node in (START, END):
            raise TypeError(f"{node} is a sentinel, not a node")
        if node in seen:
            raise TypeError(f"node {node} is listed twice")
        seen.add(node)
        nodes.append(node)

    edges: list[GraphEdge] = []
    for raw in raw_edges:
        if not isinstance(raw, dict):
            raise TypeError("each edge must be an object")
        extra = sorted(set(raw) - {"source", "target", "conditional", "label"})
        if extra:
            raise TypeError(f"unknown edge fields: {', '.join(extra)}")
        source = raw.get("source")
        target = raw.get("target")
        if not isinstance(source, str) or not source or not isinstance(target, str) or not target:
            raise TypeError("each edge needs a source and a target")
        if "conditional" in raw and not isinstance(raw["conditional"], bool):
            raise TypeError(f"edge {source} -> {target}: conditional must be a boolean")
        label = raw.get("label", "")
        if not isinstance(label, str):
            raise TypeError(f"edge {source} -> {target}: label must be a string")
        edges.append(
            GraphEdge(
                source=source,
                target=target,
                conditional=bool(raw.get("conditional", False)),
                label=label,
            )
        )
    return GraphShape(tuple(nodes), tuple(edges))


def shape_from_payload(payload: Mapping[str, Any], *, extra: tuple[str, ...] = ()) -> GraphShape:
    """Read a tool payload that is a graph document plus named extra fields."""
    if not isinstance(payload, dict):
        raise TypeError("graph payload must be an object")
    unknown = sorted(set(payload) - {"nodes", "edges", *extra})
    if unknown:
        raise TypeError(f"unknown graph fields: {', '.join(unknown)}")
    return shape_from_document({"nodes": payload.get("nodes"), "edges": payload.get("edges")})


def load_graph_document(path: Path) -> GraphShape:
    """Read a JSON graph document from disk."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise GraphDocumentError(str(exc)) from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GraphDocumentError(f"graph document is not JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise GraphDocumentError("graph document must be a JSON object")
    try:
        return shape_from_document(raw)
    except TypeError as exc:
        raise GraphDocumentError(str(exc)) from exc


def shape_from_compiled(graph: Any) -> GraphShape:
    """Read a compiled LangGraph through ``get_graph()``."""
    getter = getattr(graph, "get_graph", None)
    if not callable(getter):
        raise TypeError(
            f"{type(graph).__name__} has no get_graph(); a compiled LangGraph is what this reads"
        )
    drawable = getter()
    nodes_map = getattr(drawable, "nodes", None)
    raw_edges = getattr(drawable, "edges", None)
    if not isinstance(nodes_map, dict) or raw_edges is None:
        raise TypeError("get_graph() did not return a drawable with nodes and edges")

    node_ids: list[str] = []
    for node in nodes_map.values():
        node_id = getattr(node, "id", None)
        if not isinstance(node_id, str) or not node_id:
            raise TypeError("drawable node has no id")
        if node_id in (START, END):
            continue
        node_ids.append(node_id)

    edges: list[GraphEdge] = []
    for edge in raw_edges:
        source = getattr(edge, "source", None)
        target = getattr(edge, "target", None)
        if not isinstance(source, str) or not source or not isinstance(target, str) or not target:
            raise TypeError("drawable edge has no source and target")
        if not hasattr(edge, "conditional"):
            raise TypeError("drawable edge has no conditional flag")
        label = getattr(edge, "data", None)
        if label is None:
            label = ""
        elif not isinstance(label, str):
            raise TypeError(
                f"drawable edge {source} -> {target} has data {type(label).__name__}, "
                "which this reader does not treat as a label"
            )
        edges.append(
            GraphEdge(
                source=source,
                target=target,
                conditional=bool(edge.conditional),
                label=label,
            )
        )
    return GraphShape(tuple(node_ids), tuple(edges))


def _outgoing(shape: GraphShape) -> dict[str, list[GraphEdge]]:
    outgoing: dict[str, list[GraphEdge]] = {node: [] for node in shape.nodes}
    for edge in shape.edges:
        if edge.source in outgoing:
            outgoing[edge.source].append(edge)
    return outgoing


def _entries(shape: GraphShape) -> list[str]:
    nodes = set(shape.nodes)
    has_start = any(edge.source == START for edge in shape.edges)
    if has_start:
        entries: list[str] = []
        seen: set[str] = set()
        for edge in shape.edges:
            if edge.source == START and edge.target in nodes and edge.target not in seen:
                entries.append(edge.target)
                seen.add(edge.target)
        return entries
    incoming = {edge.target for edge in shape.edges if edge.target in nodes and edge.source in nodes}
    return [node for node in shape.nodes if node not in incoming]


def _exits(shape: GraphShape) -> list[str]:
    nodes = set(shape.nodes)
    exits: list[str] = []
    seen: set[str] = set()
    for edge in shape.edges:
        if edge.target == END and edge.source in nodes and edge.source not in seen:
            exits.append(edge.source)
            seen.add(edge.source)
    return exits


def _unreachable(shape: GraphShape, entries: list[str]) -> list[str]:
    if not entries:
        return []
    outgoing = _outgoing(shape)
    seen: set[str] = set()
    queue = list(entries)
    while queue:
        node = queue.pop(0)
        if node in seen or node == END:
            continue
        seen.add(node)
        for edge in outgoing.get(node, ()):
            if edge.target not in seen and edge.target != END:
                queue.append(edge.target)
    return [node for node in shape.nodes if node not in seen]


def _normalize_cycle(cycle: list[str]) -> tuple[str, ...]:
    body = cycle[:-1]
    if not body:
        return tuple(cycle)
    start = min(range(len(body)), key=lambda index: body[index])
    rotated = body[start:] + body[:start]
    return tuple([*rotated, rotated[0]])


def _cycles(shape: GraphShape) -> list[list[str]]:
    nodes = set(shape.nodes)
    outgoing = _outgoing(shape)
    found: list[tuple[str, ...]] = []
    seen_cycles: set[tuple[str, ...]] = set()
    visited: set[str] = set()

    def walk(node: str, stack: list[str], onstack: set[str]) -> None:
        visited.add(node)
        stack.append(node)
        onstack.add(node)
        for edge in outgoing.get(node, ()):
            nxt = edge.target
            if nxt == END or nxt not in nodes:
                continue
            if nxt in onstack:
                normalized = _normalize_cycle([*stack[stack.index(nxt) :], nxt])
                if normalized not in seen_cycles:
                    seen_cycles.add(normalized)
                    found.append(normalized)
            elif nxt not in visited:
                walk(nxt, stack, onstack)
        stack.pop()
        onstack.discard(node)

    for node in shape.nodes:
        if node not in visited:
            walk(node, [], set())
    return [list(cycle) for cycle in found]


def _unknown_nodes(shape: GraphShape) -> list[str]:
    nodes = set(shape.nodes)
    found: list[str] = []
    seen: set[str] = set()
    for edge in shape.edges:
        endpoints = (
            (edge.source, edge.source in nodes or edge.source == START),
            (edge.target, edge.target in nodes or edge.target == END),
        )
        for endpoint, known in endpoints:
            if known or endpoint in seen:
                continue
            seen.add(endpoint)
            found.append(endpoint)
    return found


def _edge_dict(edge: GraphEdge) -> dict[str, Any]:
    return {
        "source": edge.source,
        "target": edge.target,
        "conditional": edge.conditional,
        "label": edge.label,
    }


def inspect_graph(shape: GraphShape) -> dict[str, Any]:
    """Nodes, edges, entries, exits, unreachable nodes, and cycles."""
    if not isinstance(shape, GraphShape):
        raise TypeError("inspect_graph expects a GraphShape")
    entries = _entries(shape)
    return {
        "nodes": list(shape.nodes),
        "edges": [_edge_dict(edge) for edge in shape.edges],
        "entries": entries,
        "exits": _exits(shape),
        "unreachable": _unreachable(shape, entries),
        "cycles": _cycles(shape),
    }


def validate_graph(shape: GraphShape) -> dict[str, Any]:
    """``inspect_graph`` plus ``ok`` and ``findings``.

    ``ok`` is false when a finding has severity ``error``. A cycle is severity
    ``cycle`` and does not, by itself, fail the graph.
    """
    report = inspect_graph(shape)
    findings: list[dict[str, Any]] = []
    if not report["entries"]:
        findings.append({"severity": "error", "code": "no_entry", "message": "the graph has no entry"})
    for name in _unknown_nodes(shape):
        findings.append(
            {
                "severity": "error",
                "code": "unknown_node",
                "node": name,
                "message": f"{name} is not a declared node, {START}, or {END}",
            }
        )
    for name in report["unreachable"]:
        findings.append(
            {
                "severity": "error",
                "code": "unreachable",
                "node": name,
                "message": f"{name} is not reachable from an entry",
            }
        )
    outgoing = _outgoing(shape)
    for name in shape.nodes:
        if not outgoing[name]:
            findings.append(
                {
                    "severity": "error",
                    "code": "dead_end",
                    "node": name,
                    "message": f"{name} has no outgoing edge",
                }
            )
    if not report["exits"]:
        findings.append({"severity": "error", "code": "no_exit", "message": f"no edge reaches {END}"})
    for cycle in report["cycles"]:
        findings.append(
            {
                "severity": "cycle",
                "code": "cycle",
                "nodes": cycle,
                "message": "cycle " + " -> ".join(cycle),
            }
        )
    report["findings"] = findings
    report["ok"] = not any(item["severity"] == "error" for item in findings)
    return report


def _mermaid_label(text: str) -> str:
    return text.replace('"', "'").replace("\n", " ").replace("|", "/")


def _mermaid_ids(names: list[str]) -> dict[str, str]:
    used: dict[str, str] = {}
    assigned: dict[str, str] = {}
    for name in names:
        if _MERMAID_ID.match(name) and name not in used:
            assigned[name] = name
            used[name] = name
            continue
        base = re.sub(r"[^0-9A-Za-z_]", "_", name)
        if not base or base[0].isdigit():
            base = f"n_{base}"
        candidate = base
        suffix = 2
        while candidate in used and used[candidate] != name:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used[candidate] = name
        assigned[name] = candidate
    return assigned


def _node_ref(name: str, ids: dict[str, str]) -> str:
    return f'{ids[name]}["{_mermaid_label(name)}"]'


def to_mermaid(shape: GraphShape) -> str:
    """A flowchart of the shape. Conditional edges are drawn dashed."""
    if not isinstance(shape, GraphShape):
        raise TypeError("to_mermaid expects a GraphShape")
    names: list[str] = []
    if any(edge.source == START or edge.target == START for edge in shape.edges):
        names.append(START)
    names.extend(shape.nodes)
    if any(edge.source == END or edge.target == END for edge in shape.edges):
        names.append(END)
    ids = _mermaid_ids(names)
    lines = ["flowchart TD"]
    for name in names:
        lines.append(f"  {_node_ref(name, ids)}")
    for edge in shape.edges:
        if edge.source not in ids or edge.target not in ids:
            continue
        arrow = "-.->" if edge.conditional else "-->"
        if edge.label:
            arrow = f"{arrow}|{_mermaid_label(edge.label)}|"
        lines.append(f"  {_node_ref(edge.source, ids)} {arrow} {_node_ref(edge.target, ids)}")
    return "\n".join(lines) + "\n"


def _step(index: int, node: str, edge: GraphEdge, *, cycle: bool) -> dict[str, Any]:
    return {
        "index": index,
        "node": node,
        "via": edge.source,
        "kind": "conditional" if edge.conditional else "direct",
        "label": edge.label,
        "cycle": cycle,
    }


def _branch(edge: GraphEdge) -> dict[str, Any]:
    return {"label": edge.label, "target": edge.target, "conditional": edge.conditional}


def _match_choice(node: str, edges: list[GraphEdge], choice: str) -> GraphEdge:
    by_label = [edge for edge in edges if edge.label == choice]
    if len(by_label) == 1:
        return by_label[0]
    by_target = [edge for edge in edges if edge.target == choice]
    matched = len(by_label) if by_label else len(by_target)
    if len(by_label) == 0 and len(by_target) == 1:
        return by_target[0]
    raise ValueError(f"choice {choice!r} for {node} matched {matched} edges")


def dry_run(
    shape: GraphShape,
    *,
    choices: Mapping[str, str] | None = None,
    max_steps: int = 32,
) -> dict[str, Any]:
    """Walk the shape from each entry and record the steps. No node runs."""
    if not isinstance(shape, GraphShape):
        raise TypeError("dry_run expects a GraphShape")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise TypeError("max_steps must be an int >= 1")
    if choices is None:
        chosen: dict[str, str] = {}
    elif isinstance(choices, Mapping):
        chosen = {}
        for key, value in choices.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError("choices must map node names to branch labels")
            chosen[key] = value
    else:
        raise TypeError("choices must map node names to branch labels")

    outgoing = _outgoing(shape)
    traces: list[dict[str, Any]] = []

    def finish(
        steps: list[dict[str, Any]],
        ended: str,
        branches: list[dict[str, Any]] | None = None,
    ) -> None:
        traces.append({"steps": steps, "ended": ended, "branches": branches or []})

    def walk(node: str, edge: GraphEdge, steps: list[dict[str, Any]], path: tuple[str, ...]) -> None:
        if len(steps) >= max_steps:
            finish(steps, "max_steps")
            return
        cycling = node != END and node in path
        steps = [*steps, _step(len(steps), node, edge, cycle=cycling)]
        if node == END:
            finish(steps, "exit")
            return
        if cycling:
            finish(steps, "cycle")
            return
        edges = outgoing.get(node, [])
        if not edges:
            finish(steps, "dead_end")
            return
        if any(item.conditional for item in edges):
            choice = chosen.get(node)
            if choice is None:
                finish(steps, "unresolved", [_branch(item) for item in edges])
                return
            taken = _match_choice(node, edges, choice)
            walk(taken.target, taken, steps, (*path, node))
            return
        if len(edges) == 1:
            taken = edges[0]
            walk(taken.target, taken, steps, (*path, node))
            return
        for taken in edges:
            walk(taken.target, taken, steps, (*path, node))

    nodes = set(shape.nodes)
    for entry in _entries(shape):
        start_edges = [edge for edge in shape.edges if edge.source == START and edge.target == entry]
        entering = start_edges[0] if start_edges else GraphEdge(START, entry)
        if entry not in nodes:
            continue
        walk(entry, entering, [], ())

    return {"executed": False, "traces": traces}
