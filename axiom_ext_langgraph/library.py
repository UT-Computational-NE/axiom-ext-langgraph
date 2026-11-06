# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""The library that helps the toolkit.

Axiom separates two things. The toolkit is what an agent or an operator
calls. The library is the importable API that toolkit is built on, so a
LangGraph run still has one gateway, one skill registry, and one identity.

Import from here when you are constructing or testing the toolkit. The
package root exports the same symbols. Nothing here opens a second model
client, registers a skill the caller did not ask to register, or binds an
identity the caller did not pass.

Graph inspection and the approval-gated host tool live here too, and the
command and the AEOS tools call the same functions.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover — names are resolved lazily at runtime
    from axiom_ext_langgraph.actor import acting_as, current_actor
    from axiom_ext_langgraph.chat_model import AxiomChatModel, AxiomGatewayUnavailable
    from axiom_ext_langgraph.graphs import (
        GraphPort,
        LangGraphManifestError,
        read_langgraph_manifest,
        skill_from_graph,
        skills_from_langgraph_json,
    )
    from axiom_ext_langgraph.graphwork import (
        END,
        START,
        GraphShape,
        dry_run,
        inspect_graph,
        shape_from_compiled,
        shape_from_document,
        to_mermaid,
        validate_graph,
    )
    from axiom_ext_langgraph.host_tool import (
        HostToolHeld,
        decide_host_call,
        tool_from_host,
    )
    from axiom_ext_langgraph.tools.projection import (
        SkillInvocationError,
        surface_name,
        tool_from_skill,
        tools_from_registry,
    )

__all__ = [
    "END",
    "START",
    "AxiomChatModel",
    "AxiomGatewayUnavailable",
    "GraphPort",
    "GraphShape",
    "HostToolHeld",
    "LangGraphManifestError",
    "SkillInvocationError",
    "acting_as",
    "current_actor",
    "decide_host_call",
    "dry_run",
    "inspect_graph",
    "read_langgraph_manifest",
    "shape_from_compiled",
    "shape_from_document",
    "skill_from_graph",
    "skills_from_langgraph_json",
    "surface_name",
    "to_mermaid",
    "tool_from_host",
    "tool_from_skill",
    "tools_from_registry",
    "validate_graph",
]

_EXPORTS = {
    "END": "axiom_ext_langgraph.graphwork",
    "START": "axiom_ext_langgraph.graphwork",
    "AxiomChatModel": "axiom_ext_langgraph.chat_model",
    "AxiomGatewayUnavailable": "axiom_ext_langgraph.chat_model",
    "GraphPort": "axiom_ext_langgraph.graphs",
    "GraphShape": "axiom_ext_langgraph.graphwork",
    "HostToolHeld": "axiom_ext_langgraph.host_tool",
    "LangGraphManifestError": "axiom_ext_langgraph.graphs",
    "SkillInvocationError": "axiom_ext_langgraph.tools.projection",
    "acting_as": "axiom_ext_langgraph.actor",
    "current_actor": "axiom_ext_langgraph.actor",
    "decide_host_call": "axiom_ext_langgraph.host_tool",
    "dry_run": "axiom_ext_langgraph.graphwork",
    "inspect_graph": "axiom_ext_langgraph.graphwork",
    "read_langgraph_manifest": "axiom_ext_langgraph.graphs",
    "shape_from_compiled": "axiom_ext_langgraph.graphwork",
    "shape_from_document": "axiom_ext_langgraph.graphwork",
    "skill_from_graph": "axiom_ext_langgraph.graphs",
    "skills_from_langgraph_json": "axiom_ext_langgraph.graphs",
    "surface_name": "axiom_ext_langgraph.tools.projection",
    "to_mermaid": "axiom_ext_langgraph.graphwork",
    "tool_from_host": "axiom_ext_langgraph.host_tool",
    "tool_from_skill": "axiom_ext_langgraph.tools.projection",
    "tools_from_registry": "axiom_ext_langgraph.tools.projection",
    "validate_graph": "axiom_ext_langgraph.graphwork",
}


def __getattr__(name: str) -> Any:
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
