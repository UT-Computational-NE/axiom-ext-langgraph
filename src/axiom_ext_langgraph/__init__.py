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
"""

from axiom_ext_langgraph.actor import acting_as, current_actor
from axiom_ext_langgraph.chat_model import (
    ROUTING_TIER_ENV,
    AxiomChatModel,
    AxiomGatewayUnavailable,
)
from axiom_ext_langgraph.graphs import (
    GraphPort,
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

__all__ = [
    "ROUTING_TIER_ENV",
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
    "tool_from_skill",
    "tools_from_registry",
]
