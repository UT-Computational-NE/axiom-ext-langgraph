# Copyright (c) 2026 The University of Texas at Austin
# SPDX-License-Identifier: Apache-2.0

"""The worked examples, tested the way this repo tests everything.

No gateway, no network, no credentials, no VPN. Dependencies arrive as
parameters, so these run on a laptop with `langchain-core` installed and
nothing else. That property is the reason the examples are worth copying: if
learning the platform required a configured platform, nobody would learn it.

The tool-using example needs `langgraph` itself, which is an optional extra, so
those tests skip rather than error when it is absent. An expected error trains
a reader to scroll past red.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage

from axiom_ext_langgraph.examples.ask_once import ask_once


class _FakeChat:
    """Stands in for AxiomChatModel where only ``invoke`` is exercised."""

    def __init__(self, reply: str = "1.1 MW"):
        self.reply = reply
        self.calls: list[Any] = []

    def invoke(self, messages, **kwargs):
        self.calls.append(messages)
        return AIMessage(content=self.reply)


class _FakeGateway:
    """Records what reached the gateway. The repo's own testing pattern."""

    def complete_with_tools(self, **kwargs: Any):
        self.calls = getattr(self, "calls", [])
        self.calls.append(kwargs)
        return type(
            "R", (), {"success": True, "text": "ok", "provider": "fake",
                      "model": "fake", "error": None, "tool_uses": []}
        )()


def _real_model():
    """A genuine AxiomChatModel over a fake gateway.

    ``create_react_agent`` type-checks its model, so a duck-typed stand-in is
    not enough. Injecting the gateway instead keeps the test offline while the
    model under test stays the real one.
    """
    from axiom_ext_langgraph import AxiomChatModel

    return AxiomChatModel(gateway_factory=_FakeGateway)


# ---------------------------------------------------------------------------
# ask_once — the smallest real graph
# ---------------------------------------------------------------------------


def test_ask_once_returns_the_answer_text():
    chat = _FakeChat("1.1 MW")

    assert ask_once("licensed power?", principal="@nima:netl", model=chat) == "1.1 MW"
    assert chat.calls, "the model should have been asked"


def test_ask_once_runs_inside_acting_as():
    """Identity is bound around the call, not passed beside it.

    ``acting_as`` is a context manager on purpose: a graph makes several model
    calls and several tool calls, and binding at the boundary is what stops a
    multi-step run from acting as nobody halfway through. Here there is one
    call, which is exactly why it is the example to read first.
    """
    import axiom_ext_langgraph.examples.ask_once as mod

    entered: list[str] = []

    class _Ctx:
        def __init__(self, who):
            self.who = who

        def __enter__(self):
            entered.append(self.who)
            return self

        def __exit__(self, *a):
            return False

    original = mod.__dict__.get("acting_as")
    try:
        # ask_once imports lazily inside the function, so patch the source.
        import axiom_ext_langgraph as shim

        mod_acting = shim.acting_as
        shim.acting_as = lambda who: _Ctx(who)
        ask_once("q", principal="@nima:netl", model=_FakeChat())
    finally:
        shim.acting_as = mod_acting
        if original is not None:
            mod.acting_as = original

    assert entered == ["@nima:netl"]


# ---------------------------------------------------------------------------
# with_tools — a ReAct agent over registry tools
# ---------------------------------------------------------------------------

def _langgraph_available() -> bool:
    try:
        import langgraph.prebuilt  # noqa: F401
    except ImportError:
        return False
    return True


needs_langgraph = pytest.mark.skipif(
    not _langgraph_available(),
    reason="needs the optional [graph] extra: pip install 'axiom-ext-langgraph[graph]'",
)


@needs_langgraph
def test_build_agent_uses_the_registry_not_hand_written_tools(monkeypatch):
    """The difference from the upstream tutorials, asserted.

    A hand-written ``@tool`` beside a registered skill is a second tool
    registry: what the MCP surface advertises stops matching what a graph can
    call. The example must project from the registry, and must project only
    what it was asked for.
    """
    import axiom_ext_langgraph as shim
    from axiom_ext_langgraph.examples.with_tools import build_agent

    seen: dict[str, Any] = {}

    def _fake_tools_from_registry(registry, names=None, **kw):
        seen["registry"] = registry
        seen["names"] = list(names or [])
        return []

    monkeypatch.setattr(shim, "tools_from_registry", _fake_tools_from_registry)

    registry = object()
    build_agent(registry, ["telemetry.series"], model=_real_model())

    assert seen["registry"] is registry
    assert seen["names"] == ["telemetry.series"], "only the named tools, never everything"


def test_ask_reports_what_the_agent_did_not_only_what_it_said():
    """For an agent with tools, the answer is half of what happened.

    Human Intervention Burden and Recovery Efficiency are about actions. A
    transcript holding only prose cannot support them.
    """
    from axiom_ext_langgraph.examples.with_tools import ask

    class _Graph:
        def invoke(self, state):
            call = AIMessage(content="", tool_calls=[
                {"name": "telemetry.series", "args": {}, "id": "1"}
            ])
            return {"messages": [call, AIMessage(content="peak was 1.1 MW")]}

    import axiom_ext_langgraph as shim

    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    original = shim.acting_as
    shim.acting_as = lambda who: _Ctx()
    try:
        out = ask(_Graph(), "what was the peak?", principal="@nima:netl")
    finally:
        shim.acting_as = original

    assert out["answer"] == "peak was 1.1 MW"
    assert out["tools_called"] == ["telemetry.series"]
    assert out["turns"] == 2


@pytest.mark.graph
def test_the_example_prefers_the_current_agent_constructor():
    """Nobody should learn a deprecated idiom from a file meant to teach.

    ``create_react_agent`` moved to ``langchain.agents.create_agent`` in
    LangChain 1.0 and goes away in LangGraph 2.0.

    Two gates, both load-bearing, and the second was missing at first. The
    marker keeps this out of the core-only unit job, which installs neither
    backend. The ``importorskip`` is what makes the assertion below
    unconditional: the earlier version called ``_agent_constructor()`` and then
    returned early when ``langchain`` was absent, so in the one environment it
    ran in it asserted nothing, and in the environment it could not run in it
    raised ``ModuleNotFoundError`` instead of skipping.
    """
    pytest.importorskip("langchain.agents", reason="needs langchain, not just core")

    from axiom_ext_langgraph.examples.with_tools import _agent_constructor

    assert _agent_constructor().__name__ == "create_agent"
