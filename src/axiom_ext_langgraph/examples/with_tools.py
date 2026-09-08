# Copyright (c) 2026 The University of Texas at Austin
# SPDX-License-Identifier: Apache-2.0

"""A ReAct agent over platform tools — where this differs from the tutorials.

The upstream way to give an agent a tool is to write one::

    @tool
    def get_telemetry(metric: str) -> str:
        \"\"\"Fetch a metric.\"\"\"
        ...

Do that here and you have created a second tool registry. The platform already
knows what tools exist, what they take, and who may call them; a hand-written
``@tool`` beside a registered skill is a second answer to the same question, and
the two agree right up until one is changed. What the MCP surface advertises
stops matching what a graph can call, and capability projection starts lying.

So tools come from the registry instead, and the same registration that reaches
the CLI, MCP and SKILL.md reaches your graph. You get argument schemas and
descriptions for free, because a ``SkillSpec`` already declares them.

**Name the tools you want.** Projecting everything hands a graph capabilities
nobody reviewed, and a graph that silently gains a tool because an unrelated
extension was installed is a capability change that happened without a
decision.
"""

from __future__ import annotations

from typing import Any


def build_agent(
    registry: Any,
    tool_names: list[str],
    *,
    routing_tier: str = "any",
    model=None,
):
    """Build a ReAct agent whose model and tools are both the platform's.

    Three lines differ from the upstream tutorial, and nothing else does:
    ``AxiomChatModel`` for the model, ``tools_from_registry`` for the tools, and
    ``acting_as`` around the invoke (see :func:`ask`).
    """
    from axiom_ext_langgraph import AxiomChatModel, tools_from_registry

    chat = model if model is not None else AxiomChatModel(routing_tier=routing_tier)
    tools = tools_from_registry(registry, tool_names)
    return _agent_constructor()(chat, tools)


def _agent_constructor():
    """The current agent constructor, falling back to the deprecated one.

    ``create_react_agent`` moved from ``langgraph.prebuilt`` to
    ``langchain.agents.create_agent`` in LangGraph 1.0 and is slated for removal
    in 2.0. Prefer the new one so nobody learns the old idiom from this file,
    but fall back rather than hard-fail: the ``[graph]`` extra installs
    ``langgraph`` and not ``langchain``, so the new constructor is not
    guaranteed present in an environment that satisfies this package's own
    declared dependencies.

    That mismatch is worth fixing in ``pyproject.toml`` rather than papering
    over here, and this docstring is the note saying so.
    """
    try:
        from langchain.agents import create_agent

        return create_agent
    except ImportError:
        from langgraph.prebuilt import create_react_agent

        return create_react_agent


def ask(agent: Any, question: str, *, principal: str) -> dict[str, Any]:
    """Run one turn, attributed, and report what the agent said and did.

    ``acting_as`` wraps the whole invocation rather than the model call, and
    that placement is deliberate: a ReAct loop makes several model calls and
    several tool calls, and binding identity at the boundary is what stops a
    multi-step agent from acting as nobody halfway through.

    Returns both the answer and the tools actually invoked, because for an agent
    with tools "what did it say" is only half of what happened.
    """
    from axiom_ext_langgraph import acting_as

    with acting_as(principal):
        result = agent.invoke({"messages": [("user", question)]})

    messages = result.get("messages", []) if isinstance(result, dict) else []
    return {
        "answer": _final_text(messages),
        "tools_called": _tool_names(messages),
        "turns": len(messages),
    }


def _final_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content
    return ""


def _tool_names(messages: list[Any]) -> list[str]:
    names: list[str] = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name:
                names.append(name)
    return names
