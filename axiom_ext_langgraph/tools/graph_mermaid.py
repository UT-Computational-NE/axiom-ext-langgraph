# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""AEOS tool: render a graph document as a Mermaid flowchart."""

from __future__ import annotations

from typing import Any, ClassVar

from axiom_ext_langgraph.graphwork import shape_from_payload, to_mermaid

__all__ = ["GraphMermaid"]


class GraphMermaid:
    """Return the Mermaid flowchart for a graph document."""

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
        "required": ["mermaid"],
        "properties": {"mermaid": {"type": "string"}},
    }

    @staticmethod
    def invoke(payload: dict[str, Any]) -> dict[str, Any]:
        return {"mermaid": to_mermaid(shape_from_payload(payload))}
