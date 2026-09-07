# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""The chat model is the chokepoint, so these tests are mostly about what it
refuses.

A shim that quietly degrades is worse than no shim: a graph would branch on
placeholder text, an export-controlled deployment would silently fall back to a
public provider, and nobody would see either happen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from axiom_ext_langgraph.chat_model import (
    ROUTING_TIER_ENV,
    AxiomChatModel,
    AxiomGatewayUnavailable,
    _to_ai_message,
    _to_gateway_messages,
)

# --- stand-ins for the gateway's response shapes -----------------------------


@dataclass
class FakeToolUse:
    tool_id: str
    name: str
    input: dict = field(default_factory=dict)


@dataclass
class FakeResponse:
    text: str = "hello"
    tool_use: list = field(default_factory=list)
    provider: str = "test-provider"
    model: str = "test-model"
    success: bool = True
    error: str | None = None
    stop_reason: str = "end_turn"
    input_tokens: int = 11
    output_tokens: int = 7
    cache_read_tokens: int = 3


class FakeGateway:
    """Records the call so tests can assert on what reached the gateway."""

    def __init__(self, response: FakeResponse | None = None):
        self.response = response or FakeResponse()
        self.calls: list[dict[str, Any]] = []

    def complete_with_tools(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        return self.response


def _model(gateway: FakeGateway, **kwargs: Any) -> AxiomChatModel:
    return AxiomChatModel(gateway_factory=lambda: gateway, **kwargs)


# --- the chokepoint ----------------------------------------------------------


def test_a_failed_gateway_response_raises_rather_than_returning_text():
    """The gateway degrades gracefully for a CLI by returning placeholder text
    with success=False. Handing that to a graph as content would launder a
    failure into an answer the graph then branches on."""
    gateway = FakeGateway(
        FakeResponse(
            text="LLM unavailable — no providers configured.",
            success=False,
            error="No usable LLM provider",
            provider="stub",
        )
    )

    with pytest.raises(AxiomGatewayUnavailable, match="No usable LLM provider"):
        _model(gateway).invoke([HumanMessage("hi")])


def test_export_controlled_tier_reaches_the_gateway():
    gateway = FakeGateway()
    _model(gateway, routing_tier="export_controlled").invoke([HumanMessage("hi")])

    assert gateway.calls[0]["routing_tier"] == "export_controlled"


def test_routing_tier_can_be_pinned_for_the_whole_process(monkeypatch):
    monkeypatch.setenv(ROUTING_TIER_ENV, "export_controlled")
    gateway = FakeGateway()

    _model(gateway).invoke([HumanMessage("hi")])

    assert gateway.calls[0]["routing_tier"] == "export_controlled"


def test_an_explicit_tier_beats_the_environment(monkeypatch):
    monkeypatch.setenv(ROUTING_TIER_ENV, "any")
    gateway = FakeGateway()

    _model(gateway, routing_tier="export_controlled").invoke([HumanMessage("hi")])

    assert gateway.calls[0]["routing_tier"] == "export_controlled"


def test_an_unrecognised_tier_is_refused_not_defaulted():
    """Defaulting a mistyped 'export_controlled' to 'any' would send controlled
    work to a public provider."""
    with pytest.raises(ValueError, match="routing_tier"):
        AxiomChatModel(routing_tier="exportcontrolled")


def test_default_tier_is_any_when_nothing_is_configured(monkeypatch):
    monkeypatch.delenv(ROUTING_TIER_ENV, raising=False)
    assert AxiomChatModel().routing_tier == "any"


def test_streaming_raises_rather_than_pretending():
    """LangChain's default would call _generate and yield one chunk, which looks
    like streaming works."""
    gateway = FakeGateway()
    with pytest.raises(NotImplementedError, match="stream_with_tools"):
        list(_model(gateway).stream([HumanMessage("hi")]))


# --- message translation -----------------------------------------------------


def test_system_messages_are_lifted_out_of_the_turn_list():
    """The gateway takes `system` separately; leaving them inline would send a
    system prompt as user content."""
    system, history = _to_gateway_messages(
        [SystemMessage("be careful"), HumanMessage("hi")]
    )

    assert system == "be careful"
    assert history == [{"role": "user", "content": "hi"}]


def test_multiple_system_messages_are_joined():
    system, _ = _to_gateway_messages([SystemMessage("one"), SystemMessage("two")])
    assert system == "one\n\ntwo"


def test_tool_results_carry_their_call_id():
    _, history = _to_gateway_messages([ToolMessage(content="42", tool_call_id="call_1")])

    assert history == [{"role": "tool", "content": "42", "tool_call_id": "call_1"}]


def test_assistant_tool_calls_survive_the_round_trip():
    message = AIMessage(
        content="",
        tool_calls=[{"name": "lookup", "args": {"q": "x"}, "id": "call_1", "type": "tool_call"}],
    )
    _, history = _to_gateway_messages([message])

    assert history[0]["tool_calls"] == [
        {"id": "call_1", "name": "lookup", "input": {"q": "x"}}
    ]


def test_an_unknown_message_type_is_carried_not_dropped():
    """Silently discarding a turn changes what the model saw."""

    class Weird(HumanMessage):
        pass

    _, history = _to_gateway_messages([Weird(content="odd")])
    assert history and history[0]["content"] == "odd"


# --- response translation ----------------------------------------------------


def test_tool_use_blocks_become_langchain_tool_calls():
    message = _to_ai_message(
        FakeResponse(text="", tool_use=[FakeToolUse("call_9", "search", {"q": "reactor"})])
    )

    assert message.tool_calls == [
        {"name": "search", "args": {"q": "reactor"}, "id": "call_9", "type": "tool_call"}
    ]


def test_usage_and_provenance_reach_the_message():
    """Which provider answered, and what it cost, must survive into the graph —
    otherwise cost and routing are invisible to everything downstream."""
    message = _to_ai_message(FakeResponse())

    assert message.usage_metadata["input_tokens"] == 11
    assert message.usage_metadata["output_tokens"] == 7
    assert message.usage_metadata["total_tokens"] == 18
    assert message.usage_metadata["input_token_details"]["cache_read"] == 3
    assert message.response_metadata["provider"] == "test-provider"
    assert message.response_metadata["model"] == "test-model"


# --- tool binding ------------------------------------------------------------


def _echo(text: str) -> str:
    """Echo the input."""
    return text


def test_bound_tools_reach_the_gateway_in_function_calling_shape():
    gateway = FakeGateway()
    tool = StructuredTool.from_function(func=_echo, name="echo", description="Echo.")

    _model(gateway).bind_tools([tool]).invoke([HumanMessage("hi")])

    tools = gateway.calls[0]["tools"]
    assert tools[0]["function"]["name"] == "echo"


def test_binding_does_not_mutate_the_original_model():
    """Another graph node may hold the unbound model."""
    gateway = FakeGateway()
    model = _model(gateway)
    tool = StructuredTool.from_function(func=_echo, name="echo", description="Echo.")

    model.bind_tools([tool])
    model.invoke([HumanMessage("hi")])

    assert gateway.calls[0]["tools"] is None


def test_no_tools_sends_none_rather_than_an_empty_list():
    gateway = FakeGateway()
    _model(gateway).invoke([HumanMessage("hi")])

    assert gateway.calls[0]["tools"] is None
