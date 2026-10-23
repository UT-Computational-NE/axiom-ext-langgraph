# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""AEOS tool: ask the platform approval gate whether a host call may run.

This does not run the call. The LangChain wrapper in
:func:`axiom_ext_langgraph.host_tool.tool_from_host` is what runs, and only
after the same gate has approved the action. The tool and that wrapper share
:func:`~axiom_ext_langgraph.host_tool.decide_host_call`.

``bind_gate`` is how a host, or a test, points the tool at the gate already
in use. With none bound, the tool opens the platform file queue, which is
the queue that survives the call.
"""

from __future__ import annotations

from typing import Any, ClassVar

from axiom_ext_langgraph.host_tool import decide_host_call, open_platform_gate

__all__ = ["GateHostCall"]


class GateHostCall:
    """Submit a host call to the approval gate and report the decision."""

    side_effects = "mutates_state"
    idempotent = False
    _gate: ClassVar[Any] = None

    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["name"],
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "params": {"type": "object"},
            "approval_id": {"type": "string", "minLength": 1},
        },
    }

    output_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["allowed", "action_id", "status", "category", "name"],
        "properties": {
            "allowed": {"type": "boolean"},
            "action_id": {"type": "string"},
            "status": {"type": "string"},
            "category": {"type": "string"},
            "name": {"type": "string"},
        },
    }

    @classmethod
    def bind_gate(cls, gate: Any) -> None:
        """Use ``gate`` for later ``invoke`` calls. ``None`` restores the file queue."""
        cls._gate = gate

    @classmethod
    def invoke(cls, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise TypeError("payload must be an object")
        unknown = sorted(set(payload) - {"name", "params", "approval_id"})
        if unknown:
            raise TypeError(f"unknown fields: {', '.join(unknown)}")
        name = payload.get("name")
        params = payload.get("params", {})
        approval_id = payload.get("approval_id")
        gate = cls._gate if cls._gate is not None else open_platform_gate()
        return decide_host_call(gate, name, params, approval_id=approval_id).as_dict()
