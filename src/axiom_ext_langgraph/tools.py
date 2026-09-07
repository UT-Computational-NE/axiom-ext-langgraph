# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""LangChain tools generated from the Axiom skill registry.

A skill declared in an extension manifest already reaches the CLI, MCP, A2A and
SKILL.md. This module adds LangChain as a further surface over the *same*
registration, so a graph and a chat session cannot drift about what a tool is or
what it does.

The alternative — hand-written ``@tool`` functions beside the skills — is the
second tool registry that the single-registry rule exists to prevent. What the
MCP surface advertises would stop matching what a graph can actually call, and
capability projection would start lying.

Two things this respects rather than reinvents:

**Declared surfaces.** A ``SkillSpec`` may name which projections it opts into
(``cli``, ``mcp``, ``agent_tool``, ``skill_md``). That is the bounded-exposure
guard against tool explosion, and it applies here: projecting everything would
hand a graph capabilities nobody opted in for. Skills named explicitly are
projected regardless — the caller has opted in on their behalf.

**Declared inputs.** ``SkillSpec.inputs`` is a name-to-shape map. It is
documentation-grade rather than enforced, but it is enough to give a tool real
named arguments instead of one opaque blob, which is the difference between a
model calling a tool correctly and guessing.

Nothing here registers a skill. Skills are declared by the extension that owns
them; this only projects what is already registered.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import Field, create_model

__all__ = [
    "AGENT_TOOL_SURFACE",
    "SkillInvocationError",
    "tool_from_skill",
    "tools_from_registry",
]

logger = logging.getLogger(__name__)

#: The surface name a skill opts into to be projectable as an agent tool.
AGENT_TOOL_SURFACE = "agent_tool"


class SkillInvocationError(RuntimeError):
    """A skill reported failure.

    Raised rather than returned as text so a graph sees a tool error it can
    branch on, instead of a string that reads like a successful answer.
    ``SkillResult.errors`` becomes the message.
    """


def _default_context() -> Any:
    """Build a SkillContext the way the platform builds one."""
    from pathlib import Path

    from axiom.infra.skills import SkillContext, SkillRegistry

    return SkillContext(
        registry=SkillRegistry(),
        state_dir=Path.home() / ".axi",
        logger=logging.getLogger("axiom_ext_langgraph.tools"),
        user_prompt=None,
    )


def _args_model(name: str, inputs: dict[str, str] | None) -> type | None:
    """Build a pydantic args model from a skill's declared inputs.

    ``inputs`` maps an argument name to a shape string such as ``"Path"``. The
    shapes are documentation-grade and not enforced by the platform, so every
    field is typed ``str`` here with the declared shape carried in the
    description. Naming the arguments is what matters: a tool whose only
    parameter is an untyped blob makes a model guess.
    """
    if not inputs:
        return None

    fields = {
        arg: (str, Field(description=f"{shape}" if shape else arg))
        for arg, shape in inputs.items()
    }
    return create_model(f"{name.replace('.', '_')}_Args", **fields)


def tool_from_skill(
    registry: Any,
    name: str,
    *,
    description: str = "",
    args_schema: type | None = None,
    context_factory: Callable[[], Any] | None = None,
) -> StructuredTool:
    """Project one registered skill as a LangChain tool.

    Args:
        registry: an ``axiom.infra.skills.SkillRegistry``.
        name: the skill's qualified name, e.g. ``"data.install"``.
        description: overrides the skill's declared description.
        args_schema: overrides the schema derived from the skill's declared
            inputs. Supply one when the skill predates declared inputs.
        context_factory: builds the ``SkillContext`` per invocation.

    Returns:
        A ``StructuredTool`` that invokes the skill through the registry.
    """
    build_context = context_factory or _default_context
    spec = registry.spec(name) if hasattr(registry, "spec") else None

    schema = args_schema or _args_model(name, getattr(spec, "inputs", None))

    def _invoke(**params: Any) -> Any:
        result = registry.invoke(name, params, build_context())
        if not getattr(result, "ok", False):
            errors = getattr(result, "errors", None) or ["skill reported failure"]
            raise SkillInvocationError(f"{name}: {'; '.join(str(e) for e in errors)}")
        return result.value

    kwargs: dict[str, Any] = {}
    if schema is not None:
        kwargs["args_schema"] = schema

    return StructuredTool.from_function(
        func=_invoke,
        name=name.replace(".", "_"),
        description=description or getattr(spec, "description", "") or f"Axiom skill {name}",
        **kwargs,
    )


def tools_from_registry(
    registry: Any,
    names: Iterable[str] | None = None,
    *,
    context_factory: Callable[[], Any] | None = None,
) -> list[StructuredTool]:
    """Project registered skills as LangChain tools.

    Args:
        registry: an ``axiom.infra.skills.SkillRegistry``.
        names: which skills to project, in order. Naming them is preferred: a
            graph that silently gains a tool because an unrelated extension was
            installed is a capability change nobody reviewed. Named skills are
            projected whether or not they declare the agent-tool surface — the
            caller has opted in for them.
        context_factory: builds the ``SkillContext`` per invocation.

    Returns:
        One tool per projected skill.

    Raises:
        TypeError: when ``names`` is omitted and the registry exposes no way to
            list skills.
    """
    if names is None:
        lister = getattr(registry, "list", None)
        if lister is None:
            raise TypeError(
                "registry exposes no `list()`; pass `names` explicitly rather "
                "than guessing at its API"
            )
        names = [n for n in lister() if _opts_into_agent_tools(registry, n)]

    return [
        tool_from_skill(registry, name, context_factory=context_factory) for name in names
    ]


def _opts_into_agent_tools(registry: Any, name: str) -> bool:
    """Whether a skill declared the agent-tool surface.

    Undeclared surfaces mean "not opted in" for bulk projection, matching the
    MCP server's bounded-exposure guard. A skill that wants to reach graphs says
    so; one that has never considered the question is not volunteered.
    """
    spec = registry.spec(name) if hasattr(registry, "spec") else None
    surfaces = getattr(spec, "surfaces", None)
    if not surfaces:
        logger.debug("skill %s declares no surfaces; not projected in bulk", name)
        return False
    return AGENT_TOOL_SURFACE in surfaces
