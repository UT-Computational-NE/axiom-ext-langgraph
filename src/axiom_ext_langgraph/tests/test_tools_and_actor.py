# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""Tool projection and actor binding.

Both modules take their dependencies as parameters — the registry, the context
factory — so they are testable without installing the platform. That is a
deliberate property: a shim whose tests need the whole world stops being run.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any

import pytest

from axiom_ext_langgraph.actor import acting_as, current_actor
from axiom_ext_langgraph.tools import (
    SkillInvocationError,
    tool_from_skill,
    tools_from_registry,
)

# --- tool projection ---------------------------------------------------------


@dataclass
class FakeResult:
    ok: bool = True
    value: Any = "done"
    errors: list = field(default_factory=list)


@dataclass
class FakeSpec:
    name: str
    description: str = ""
    inputs: dict = field(default_factory=dict)
    surfaces: tuple = None


class FakeRegistry:
    """Mirrors the parts of SkillRegistry this module actually uses."""

    def __init__(
        self,
        result: FakeResult | None = None,
        specs: dict[str, FakeSpec] | None = None,
    ):
        self.result = result or FakeResult()
        self._specs = specs if specs is not None else {
            "data.install": FakeSpec(
                "data.install",
                description="Install a dataset.",
                inputs={"target": "Path"},
                surfaces=("cli", "agent_tool"),
            ),
            "telemetry.series": FakeSpec(
                "telemetry.series",
                description="Fetch a series.",
                inputs={"signal": "str"},
                surfaces=("cli", "mcp", "agent_tool"),
            ),
        }
        self.calls: list[tuple] = []

    def invoke(self, name, params, ctx):
        self.calls.append((name, params, ctx))
        return self.result

    def spec(self, name):
        return self._specs.get(name)

    def list(self, namespace=None):
        return sorted(self._specs)


def _ctx():
    return object()


def test_a_projected_skill_invokes_through_the_registry():
    registry = FakeRegistry()
    tool = tool_from_skill(registry, "data.install", context_factory=_ctx)

    assert tool.invoke({"target": "x"}) == "done"
    assert registry.calls[0][0] == "data.install"
    assert registry.calls[0][1] == {"target": "x"}


def test_a_failed_skill_raises_rather_than_returning_its_error_as_text():
    """A graph must see a tool error it can branch on, not a string that reads
    like a successful answer."""
    registry = FakeRegistry(FakeResult(ok=False, errors=["disk full", "retry later"]))
    tool = tool_from_skill(registry, "data.install", context_factory=_ctx)

    with pytest.raises(SkillInvocationError, match="disk full; retry later"):
        tool.invoke({"target": "x"})


def test_a_failed_skill_with_no_message_still_raises():
    registry = FakeRegistry(FakeResult(ok=False, errors=[]))
    tool = tool_from_skill(registry, "data.install", context_factory=_ctx)

    with pytest.raises(SkillInvocationError, match="reported failure"):
        tool.invoke({"target": "x"})


def test_tool_names_are_langchain_safe():
    """Dots are not valid in a function-calling tool name."""
    tool = tool_from_skill(FakeRegistry(), "data.install", context_factory=_ctx)
    assert tool.name == "data_install"


def test_projection_preserves_the_order_it_was_given():
    registry = FakeRegistry()
    tools = tools_from_registry(
        registry, ["telemetry.series", "data.install"], context_factory=_ctx
    )
    assert [t.name for t in tools] == ["telemetry_series", "data_install"]


def test_projecting_everything_uses_the_registry_listing():
    registry = FakeRegistry()
    tools = tools_from_registry(registry, context_factory=_ctx)
    assert [t.name for t in tools] == ["data_install", "telemetry_series"]


def test_bulk_projection_skips_skills_that_did_not_opt_into_agent_tools():
    """Declared surfaces are the bounded-exposure guard against tool explosion.
    Projecting everything would hand a graph capabilities nobody opted in for."""
    registry = FakeRegistry(
        specs={
            "data.install": FakeSpec("data.install", surfaces=("cli", "agent_tool")),
            "secret.rotate": FakeSpec("secret.rotate", surfaces=("cli",)),
        }
    )
    tools = tools_from_registry(registry, context_factory=_ctx)

    assert [t.name for t in tools] == ["data_install"]


def test_a_skill_with_undeclared_surfaces_is_not_volunteered():
    """Undeclared means the question was never considered, not yes."""
    registry = FakeRegistry(specs={"legacy.thing": FakeSpec("legacy.thing")})
    assert tools_from_registry(registry, context_factory=_ctx) == []


def test_naming_a_skill_projects_it_regardless_of_declared_surfaces():
    """The caller has opted in on its behalf."""
    registry = FakeRegistry(specs={"secret.rotate": FakeSpec("secret.rotate", surfaces=("cli",))})
    tools = tools_from_registry(registry, ["secret.rotate"], context_factory=_ctx)

    assert [t.name for t in tools] == ["secret_rotate"]


def test_declared_inputs_become_named_tool_arguments():
    """A tool whose only parameter is an untyped blob makes a model guess."""
    tool = tool_from_skill(FakeRegistry(), "data.install", context_factory=_ctx)
    assert "target" in tool.args


def test_the_skills_own_description_is_used():
    tool = tool_from_skill(FakeRegistry(), "data.install", context_factory=_ctx)
    assert tool.description == "Install a dataset."


def test_a_registry_with_no_listing_api_is_refused_rather_than_guessed_at():
    """Guessing at a second method name is how a bridge ends up calling
    something that does not exist and returning nothing in silence."""

    class Opaque:
        def invoke(self, name, params, ctx):
            return FakeResult()

    with pytest.raises(TypeError, match="pass `names` explicitly"):
        tools_from_registry(Opaque(), context_factory=_ctx)


# --- actor binding -----------------------------------------------------------


@pytest.fixture
def fake_governance(monkeypatch):
    """Stand in for axiom.governance so the shim is testable without it."""
    module = types.ModuleType("axiom.governance")
    state: dict[str, Any] = {"actor": None}

    def get_current_actor(**_kwargs):
        if state["actor"] is None:
            raise RuntimeError("no actor bound")
        return state["actor"]

    def set_current_actor(principal):
        state["actor"] = principal

    module.get_current_actor = get_current_actor
    module.set_current_actor = set_current_actor

    axiom_pkg = sys.modules.get("axiom") or types.ModuleType("axiom")
    monkeypatch.setitem(sys.modules, "axiom", axiom_pkg)
    monkeypatch.setitem(sys.modules, "axiom.governance", module)
    return state


def test_acting_as_binds_the_principal(fake_governance):
    with acting_as("@ben:ut") as bound:
        assert bound == "@ben:ut"
        assert current_actor() == "@ben:ut"


def test_the_previous_actor_is_restored(fake_governance):
    fake_governance["actor"] = "@outer:ut"

    with acting_as("@inner:ut"):
        assert current_actor() == "@inner:ut"

    assert current_actor() == "@outer:ut"


def test_the_previous_actor_is_restored_even_when_the_run_raises(fake_governance):
    """A failed graph must not leave someone else's principal bound."""
    fake_governance["actor"] = "@outer:ut"

    with pytest.raises(ValueError), acting_as("@inner:ut"):
        raise ValueError("graph blew up")

    assert current_actor() == "@outer:ut"


def test_current_actor_is_none_when_nothing_is_bound(fake_governance):
    assert current_actor() is None
