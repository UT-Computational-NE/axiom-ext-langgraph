# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""The LangGraph toolkit.

In Axiom's vocabulary a toolkit is the capability surface an agent or an
operator invokes: here, LangChain tools projected from the one skill
registry (:class:`AxiomToolkit`), the ``project_skills`` tool, the
``langgraph port`` command, and the ``langgraph.report`` skill.

The library that makes this toolkit a client of the platform, rather than
a second registry, lives in :mod:`axiom_ext_langgraph.library`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from langchain_core.tools import BaseToolkit
from pydantic import ConfigDict, Field

from axiom_ext_langgraph.tools.project_skills import ProjectSkills
from axiom_ext_langgraph.tools.projection import tools_from_registry

__all__ = ["AxiomToolkit", "ProjectSkills"]


class AxiomToolkit(BaseToolkit):
    """LangChain toolkit over the one Axiom skill registry.

    Hand ``get_tools()`` to ``create_agent``. Naming skills is preferred:
    a graph that silently gains a tool because another extension was
    installed is a capability change nobody reviewed. Named skills are
    projected whether or not they declare the agent-tool surface — the
    caller has opted in. With ``names`` left unset, only skills that
    opted into ``agent_tool`` are projected.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    registry: Any
    names: list[str] | None = None
    context_factory: Callable[[], Any] | None = Field(default=None, exclude=True)

    def get_tools(self) -> list[Any]:
        selected: Iterable[str] | None = self.names
        return tools_from_registry(
            self.registry,
            selected,
            context_factory=self.context_factory,
        )
