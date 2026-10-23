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
    from axiom_ext_langgraph.tools.projection import (
        SkillInvocationError,
        surface_name,
        tool_from_skill,
        tools_from_registry,
    )

__all__ = [
    "AxiomChatModel",
    "AxiomGatewayUnavailable",
    "GraphPort",
    "LangGraphManifestError",
    "SkillInvocationError",
    "acting_as",
    "current_actor",
    "read_langgraph_manifest",
    "skill_from_graph",
    "skills_from_langgraph_json",
    "surface_name",
    "tool_from_skill",
    "tools_from_registry",
]

_EXPORTS = {
    "AxiomChatModel": "axiom_ext_langgraph.chat_model",
    "AxiomGatewayUnavailable": "axiom_ext_langgraph.chat_model",
    "GraphPort": "axiom_ext_langgraph.graphs",
    "LangGraphManifestError": "axiom_ext_langgraph.graphs",
    "SkillInvocationError": "axiom_ext_langgraph.tools.projection",
    "acting_as": "axiom_ext_langgraph.actor",
    "current_actor": "axiom_ext_langgraph.actor",
    "read_langgraph_manifest": "axiom_ext_langgraph.graphs",
    "skill_from_graph": "axiom_ext_langgraph.graphs",
    "skills_from_langgraph_json": "axiom_ext_langgraph.graphs",
    "surface_name": "axiom_ext_langgraph.tools.projection",
    "tool_from_skill": "axiom_ext_langgraph.tools.projection",
    "tools_from_registry": "axiom_ext_langgraph.tools.projection",
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
