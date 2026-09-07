# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""A real LangGraph, driven by the shim, against a stub gateway.

Every other test here exercises the shim against ``langchain-core`` alone. That
proves the translation is right but not that a *graph* can actually be built on
it — a compiled graph calls methods, inspects attributes and threads state in
ways a direct call does not.

The gateway is stubbed rather than live: the point is that the contract holds,
not that a provider answers. What is deliberately real is LangGraph itself — the
graph is built from ``StateGraph``, ``ToolNode`` and ``tools_condition``, the
primitives that are not migrating between packages.

Marked ``graph`` so the default suite stays runnable with only langchain-core
installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

import pytest

from axiom_ext_langgraph import AxiomChatModel, AxiomGatewayUnavailable

pytest.importorskip("langgraph", reason="langgraph is an optional extra")

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

pytestmark = pytest.mark.graph


class _State(TypedDict):
    messages: Annotated[list, add_messages]


def build_agent(model: AxiomChatModel, tools: list):
    """The canonical tool-calling loop, wired explicitly.

    Written out rather than pulled from a prebuilt helper for two reasons: the
    prebuilt one is migrating between packages, and this is the reference a
    reader copies — showing the wiring is more useful than hiding it.
    """
    bound = model.bind_tools(tools)

    def call_model(state: _State) -> dict:
        return {"messages": [bound.invoke(state["messages"])]}

    graph = StateGraph(_State)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")
    return graph.compile()


@dataclass
class StubToolUse:
    tool_id: str
    name: str
    input: dict = field(default_factory=dict)


@dataclass
class StubResponse:
    text: str = ""
    tool_use: list = field(default_factory=list)
    provider: str = "stub-provider"
    model: str = "stub-model"
    success: bool = True
    error: str | None = None
    stop_reason: str = "end_turn"
    input_tokens: int = 5
    output_tokens: int = 3
    cache_read_tokens: int = 0


class ScriptedGateway:
    """Returns each scripted response in turn, so a tool loop can be driven."""

    def __init__(self, *responses: StubResponse):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete_with_tools(self, **kwargs: Any) -> StubResponse:
        self.calls.append(kwargs)
        if not self.responses:
            return StubResponse(text="done")
        return self.responses.pop(0)


def _reactor_power(unit: str) -> str:
    """Return the current reactor power in the requested unit."""
    return f"950 {unit}"


def test_a_real_graph_builds_and_runs_on_the_shim():
    """The contract check: a real compiled LangGraph runs against the shim."""
    gateway = ScriptedGateway(StubResponse(text="Power is 950 kW."))
    model = AxiomChatModel(gateway_factory=lambda: gateway)

    agent = build_agent(model, [_reactor_power])
    result = agent.invoke({"messages": [("user", "what is reactor power?")]})

    assert result["messages"][-1].content == "Power is 950 kW."
    assert gateway.calls, "the graph never reached the gateway"


def test_a_tool_call_round_trips_through_the_graph():
    """The model asks for a tool, LangGraph runs it, the result comes back as a
    tool message, and the second turn sees it. This is the path where a wrong
    message translation shows up as a graph that loops or stalls."""
    gateway = ScriptedGateway(
        StubResponse(tool_use=[StubToolUse("call_1", "_reactor_power", {"unit": "kW"})]),
        StubResponse(text="Power is 950 kW."),
    )
    model = AxiomChatModel(gateway_factory=lambda: gateway)

    agent = build_agent(model, [_reactor_power])
    result = agent.invoke({"messages": [("user", "power?")]})

    assert result["messages"][-1].content == "Power is 950 kW."

    # The second gateway call must carry the tool result, or the model answered
    # without ever seeing what the tool returned.
    second_turn = gateway.calls[1]["messages"]
    assert any(m.get("role") == "tool" and "950" in str(m.get("content")) for m in second_turn)


def test_every_call_in_the_graph_carries_the_routing_tier():
    """A graph must not be able to reach a provider outside its pinned tier,
    including on turns the caller never sees."""
    gateway = ScriptedGateway(
        StubResponse(tool_use=[StubToolUse("call_1", "_reactor_power", {"unit": "kW"})]),
        StubResponse(text="done"),
    )
    model = AxiomChatModel(
        gateway_factory=lambda: gateway, routing_tier="export_controlled"
    )

    build_agent(model, [_reactor_power]).invoke({"messages": [("user", "power?")]})

    assert len(gateway.calls) >= 2
    assert all(call["routing_tier"] == "export_controlled" for call in gateway.calls)


def test_a_gateway_failure_halts_the_graph():
    """Rather than the graph continuing on placeholder text."""
    gateway = ScriptedGateway(
        StubResponse(
            text="LLM unavailable — no providers configured.",
            success=False,
            error="No usable LLM provider",
        )
    )
    model = AxiomChatModel(gateway_factory=lambda: gateway)

    with pytest.raises(AxiomGatewayUnavailable):
        build_agent(model, [_reactor_power]).invoke({"messages": [("user", "power?")]})


def test_tools_reach_the_gateway_in_function_calling_shape():
    """LangGraph binds the tools; the shim must convert what it is handed."""
    gateway = ScriptedGateway(StubResponse(text="ok"))
    model = AxiomChatModel(gateway_factory=lambda: gateway)

    build_agent(model, [_reactor_power]).invoke({"messages": [("user", "hi")]})

    tools = gateway.calls[0]["tools"]
    assert tools and tools[0]["function"]["name"] == "_reactor_power"
