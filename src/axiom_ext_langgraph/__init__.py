# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""Run LangChain and LangGraph through the Axiom gateway.

Hosting a foreign agent runtime is fine. Hosting a second *brain* is not. The
difference is whether there stays exactly one of each of four things:

1. **one gateway** — :mod:`~axiom_ext_langgraph.chat_model`
2. **one state and audit record** — not implemented here; see AGENTS.md
3. **one tool registry** — :mod:`~axiom_ext_langgraph.tools`
4. **one identity** — :mod:`~axiom_ext_langgraph.actor`

With those single, a LangGraph run is a rendering of the platform's behaviour.
With any doubled, the split is real and the framework's name is irrelevant.

Imports are lazy, and that is load-bearing rather than tidy.

A ported extension that only registers graphs still has to import this package
to reach :func:`~axiom_ext_langgraph.graphs.skills_from_langgraph_json`. Eager
imports made that pull in :mod:`~axiom_ext_langgraph.chat_model`, which needs
``langchain_core``, which a graph-only extension may not have installed. The
platform's extension loader swallows an import error and returns no
capabilities, so the whole port would fail with no message at all.

``graphs`` needs nothing but the standard library and Axiom. It stays reachable
on its own.

One import stays eager, and must: external telemetry. The tracing guard's
module body strips LangChain-family configuration from the process, and it has
to run before anything that reads it. Laziness everywhere else makes that
easier rather than harder — with no sibling imported at load time, the guard is
the only thing that has run when it does its work.
"""

import importlib
from typing import TYPE_CHECKING

# Eager, and first. Its module body strips LangChain-family env config before
# any sibling can import langchain and read it. Everything else below is lazy,
# which means nothing else has been imported by the time this lands.
from axiom_ext_langgraph._tracing_guard import enforce_no_external_tracing

_LAZY = {
    "ROUTING_TIER_ENV": "axiom_ext_langgraph.chat_model",
    "AxiomChatModel": "axiom_ext_langgraph.chat_model",
    "AxiomGatewayUnavailable": "axiom_ext_langgraph.chat_model",
    "acting_as": "axiom_ext_langgraph.actor",
    "current_actor": "axiom_ext_langgraph.actor",
    "GraphPort": "axiom_ext_langgraph.graphs",
    "GraphResolutionError": "axiom_ext_langgraph.graphs",
    "LangGraphManifestError": "axiom_ext_langgraph.graphs",
    "read_langgraph_manifest": "axiom_ext_langgraph.graphs",
    "skill_from_graph": "axiom_ext_langgraph.graphs",
    "skills_from_langgraph_json": "axiom_ext_langgraph.graphs",
    "SkillInvocationError": "axiom_ext_langgraph.tools",
    "tool_from_skill": "axiom_ext_langgraph.tools",
    "tools_from_registry": "axiom_ext_langgraph.tools",
}

if TYPE_CHECKING:  # pragma: no cover — for type checkers and editors only
    from axiom_ext_langgraph.actor import acting_as, current_actor
    from axiom_ext_langgraph.chat_model import (
        ROUTING_TIER_ENV,
        AxiomChatModel,
        AxiomGatewayUnavailable,
    )
    from axiom_ext_langgraph.graphs import (
        GraphPort,
        GraphResolutionError,
        LangGraphManifestError,
        read_langgraph_manifest,
        skill_from_graph,
        skills_from_langgraph_json,
    )
    from axiom_ext_langgraph.tools import (
        SkillInvocationError,
        tool_from_skill,
        tools_from_registry,
    )


def __getattr__(name: str):
    """PEP 562 — resolve a public name on first use, then cache it."""
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "ROUTING_TIER_ENV",
    "AxiomChatModel",
    "AxiomGatewayUnavailable",
    "GraphPort",
    "GraphResolutionError",
    "LangGraphManifestError",
    "SkillInvocationError",
    "acting_as",
    "current_actor",
    "enforce_no_external_tracing",
    "read_langgraph_manifest",
    "skill_from_graph",
    "skills_from_langgraph_json",
    "tool_from_skill",
    "tools_from_registry",
]
