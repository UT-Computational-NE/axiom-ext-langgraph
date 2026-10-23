# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""AEOS tool: inspect a graph document. No model call, no node execution."""

from __future__ import annotations

from typing import Any, ClassVar

from axiom_ext_langgraph.graphwork import inspect_graph, shape_from_payload

__all__ = ["InspectGraph"]


class InspectGraph:
    """Return nodes, edges, entries, exits, unreachable nodes, and cycles."""

    side_effects = "none"
    idempotent = True

    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["nodes", "edges"],
        "properties": {
            "nodes": {"type": "array", "items": {"type": "string", "minLength": 1}},
            "edges": {"type": "array", "items": {"type": "object"}},
        },
    }

    output_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["nodes", "edges", "entries", "exits", "unreachable", "cycles"],
        "properties": {
            "nodes": {"type": "array"},
            "edges": {"type": "array"},
            "entries": {"type": "array"},
            "exits": {"type": "array"},
            "unreachable": {"type": "array"},
            "cycles": {"type": "array"},
        },
    }

    @staticmethod
    def invoke(payload: dict[str, Any]) -> dict[str, Any]:
        return inspect_graph(shape_from_payload(payload))
