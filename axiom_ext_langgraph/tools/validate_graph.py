# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""AEOS tool: validate a graph document. Same report as inspect, plus findings."""

from __future__ import annotations

from typing import Any, ClassVar

from axiom_ext_langgraph.graphwork import shape_from_payload, validate_graph

__all__ = ["ValidateGraph"]


class ValidateGraph:
    """Report whether the graph is structurally sound, and why."""

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
        "required": ["ok", "findings", "nodes", "unreachable", "cycles"],
        "properties": {
            "ok": {"type": "boolean"},
            "findings": {"type": "array"},
            "nodes": {"type": "array"},
            "unreachable": {"type": "array"},
            "cycles": {"type": "array"},
        },
    }

    @staticmethod
    def invoke(payload: dict[str, Any]) -> dict[str, Any]:
        return validate_graph(shape_from_payload(payload))
