# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""AEOS tool: dry-run a graph document and return the traced steps."""

from __future__ import annotations

from typing import Any, ClassVar

from axiom_ext_langgraph.graphwork import dry_run, shape_from_payload

__all__ = ["DryRunGraph"]


class DryRunGraph:
    """Walk edges and record steps. Node functions are not called."""

    side_effects = "none"
    idempotent = True

    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["nodes", "edges"],
        "properties": {
            "nodes": {"type": "array", "items": {"type": "string", "minLength": 1}},
            "edges": {"type": "array", "items": {"type": "object"}},
            "choices": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": "Map a node name to the branch label to follow.",
            },
        },
    }

    output_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["executed", "traces"],
        "properties": {
            "executed": {"type": "boolean"},
            "traces": {"type": "array"},
        },
    }

    @staticmethod
    def invoke(payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise TypeError("graph payload must be an object")
        choices = payload.get("choices")
        shape = shape_from_payload(payload, extra=("choices",))
        return dry_run(shape, choices=choices)
