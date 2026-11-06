# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""The AEOS tool for the LangGraph toolkit.

:class:`AxiomToolkit` projects a live registry into LangChain tools. This
class is the same projection with the registry left outside the call: the
caller passes each skill's declared name, description, and inputs, and gets
back the surface-safe tool specs. It does not register a skill, call a
model, or read a file.
"""

from __future__ import annotations

from typing import Any, ClassVar

from axiom_ext_langgraph.tools.projection import surface_name

__all__ = ["ProjectSkills"]


class ProjectSkills:
    """Project declared skills into LangChain-facing tool specs.

    ``invoke`` takes one dict, which is the shape an MCP caller uses. The
    schemas below are the typed arguments AEOS §4.2 requires a tool to have.
    """

    side_effects = "none"
    idempotent = True

    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["skills"],
        "properties": {
            "skills": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name"],
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "description": {"type": "string"},
                        "inputs": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                        },
                    },
                },
            }
        },
    }

    output_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["tools"],
        "properties": {
            "tools": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "description", "arguments"],
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "arguments": {"type": "array"},
                    },
                },
            }
        },
    }

    @staticmethod
    def invoke(payload: dict[str, Any]) -> dict[str, Any]:
        """Return one tool spec per declared skill, in the given order."""
        skills = payload.get("skills") if isinstance(payload, dict) else None
        if not isinstance(skills, list):
            raise TypeError("skills must be a list of {name, description, inputs} objects")

        tools: list[dict[str, Any]] = []
        for item in skills:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"]:
                raise TypeError("each skill needs a non-empty name")
            name = item["name"]
            inputs = item.get("inputs") or {}
            if not isinstance(inputs, dict):
                raise TypeError(f"{name}: inputs must be a name-to-shape map")
            tools.append(
                {
                    "name": surface_name(name),
                    "description": item.get("description") or f"Axiom skill {name}",
                    "arguments": [
                        {"name": arg, "description": str(shape) if shape else arg}
                        for arg, shape in inputs.items()
                    ],
                }
            )
        return {"tools": tools}
