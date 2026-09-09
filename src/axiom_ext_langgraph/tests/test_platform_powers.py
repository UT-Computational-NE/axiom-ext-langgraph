# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""What a ported graph gains: the platform's capabilities, as its own tools.

The port so far argues that nothing is lost. This is the other half. A graph
that becomes an Axiom capability can also *call* Axiom capabilities, and it
gets them the same way it got its own surfaces — by declaration, with no
adapter written for any of them.

The number available today is small and that is not the claim. The claim is the
mechanism: every capability the platform declares with ``agent_tool`` is
already a tool this graph can call, so the catalogue grows without the graph
changing.
"""

from __future__ import annotations

import logging

import pytest

from axiom_ext_langgraph.tools import tools_from_registry


def _needs_principal_conversion():
    """Skip unless the platform can turn a handle into a bindable Principal.

    ``acting_as`` converts through ``axiom.governance.principal_from_handle``.
    Where that is absent — a platform release predating it — the shim falls
    back to binding what it was handed, ``get_current_actor`` refuses a
    non-Principal, and the handle reads back as None.

    That fallback is deliberate and keeps an older platform working, but it
    means these assertions describe a capability the environment may not have.
    Skipping names the missing piece; failing would report the shim as broken
    when it is the platform that is older.
    """
    governance = pytest.importorskip("axiom.governance", reason="needs the platform")
    if not hasattr(governance, "principal_from_handle"):
        pytest.skip(
            "this platform has no governance.principal_from_handle, so a handle "
            "cannot be converted into a bindable Principal (axiom "
            "feat/durable-approval-gate adds it)"
        )
    return governance


@pytest.fixture
def platform():
    """A registry with a read verb, a write verb, and one that is CLI-only."""
    skills = pytest.importorskip("axiom.infra.skills", reason="needs the platform")
    calls: list[dict] = []

    def check_deck(params, ctx):
        calls.append(dict(params))
        return skills.SkillResult(
            ok=True, value={"verdict": "pass", "mesh": params.get("mesh")}
        )

    def promote(params, ctx):
        calls.append(dict(params))
        return skills.SkillResult(ok=True, value={"promoted": params.get("model_ref")})

    def approve(params, ctx):  # pragma: no cover — must never be reachable
        raise AssertionError("a CLI-only verb must not be offered to a graph")

    registry = skills.SkillRegistry()
    registry.register_skill(
        skills.SkillSpec(
            name="corral.check_deck",
            fn=check_deck,
            description="Validate a generated deck against the corral's checks.",
            inputs={"mesh": "str"},
            side_effects=False,
            idempotent=True,
            surfaces=("cli", "mcp", "agent_tool", "skill_md"),
        )
    )
    registry.register_skill(
        skills.SkillSpec(
            name="corral.promote",
            fn=promote,
            description="Promote a model revision to serving.",
            inputs={"model_ref": "str"},
            side_effects=True,
            surfaces=("cli", "mcp", "agent_tool", "skill_md"),
        )
    )
    registry.register_skill(
        skills.SkillSpec(
            name="approval.approve",
            fn=approve,
            description="Approve a held action.",
            inputs={"action_id": "str"},
            side_effects=True,
            surfaces=("cli", "skill_md"),
        )
    )
    return registry, calls


class TestTheGraphGainsThePlatformsCapabilities:
    def test_declared_capabilities_arrive_as_tools(self, platform):
        registry, _ = platform

        names = sorted(t.name for t in tools_from_registry(registry))

        assert "corral_check_deck" in names
        assert "corral_promote" in names

    def test_a_tool_carries_the_description_a_model_decides_on(self, platform):
        """Written once on the SkillSpec, read by the CLI, MCP and the model."""
        registry, _ = platform
        tool = next(
            t for t in tools_from_registry(registry) if t.name == "corral_check_deck"
        )

        assert "corral's checks" in tool.description

    def test_a_tool_has_real_named_arguments(self, platform):
        """``SkillSpec.inputs`` is what stops a model guessing at a blob."""
        registry, _ = platform
        tool = next(
            t for t in tools_from_registry(registry) if t.name == "corral_check_deck"
        )

        assert "mesh" in tool.args_schema.model_json_schema()["properties"]

    def test_calling_the_tool_runs_the_platform_skill(self, platform):
        registry, calls = platform
        tool = next(
            t for t in tools_from_registry(registry) if t.name == "corral_check_deck"
        )

        result = tool.invoke({"mesh": "core.e"})

        assert calls == [{"mesh": "core.e"}]
        assert result["verdict"] == "pass"


class TestTheGuardHoldsHere_Too:
    """A graph must not reach a verb withheld from agents on purpose.

    ``approval.approve`` is CLI-only so an agent cannot approve its own writes.
    That decision has to survive the trip into a foreign runtime, or it was
    never a control.
    """

    def test_a_cli_only_verb_is_not_offered(self, platform):
        registry, _ = platform

        assert "approval_approve" not in {t.name for t in tools_from_registry(registry)}

    def test_naming_it_explicitly_is_the_caller_opting_in(self, platform):
        """Bulk projection honours the declaration; an explicit ask overrides it.

        Worth pinning as the deliberate escape hatch it is, so nobody later
        reads the omission above as an accident and 'fixes' it.
        """
        registry, _ = platform

        offered = tools_from_registry(registry, ["approval.approve"])

        assert [t.name for t in offered] == ["approval_approve"]


class TestIdentityReachesTheToolCall:
    def test_the_platform_skill_sees_who_is_acting(self, platform):
        """The graph's principal has to survive into the capability it calls,
        or the audit record names the process instead of the person."""
        skills = pytest.importorskip("axiom.infra.skills")
        _needs_principal_conversion()
        pc = pytest.importorskip("axiom.infra.principal")
        from axiom_ext_langgraph.actor import acting_as
        from axiom_ext_langgraph.chat_model import _current_principal_handle

        seen: dict[str, str | None] = {}
        registry = skills.SkillRegistry()
        registry.register_skill(
            skills.SkillSpec(
                name="corral.who",
                fn=lambda p, c: (
                    seen.__setitem__("handle", _current_principal_handle()),
                    skills.SkillResult(ok=True, value={}),
                )[1],
                description="Reports the acting principal.",
                inputs={},
                surfaces=("agent_tool",),
            )
        )

        def ctx_factory():
            from pathlib import Path

            return skills.SkillContext(
                registry=registry,
                state_dir=Path("/tmp"),
                logger=logging.getLogger("t"),
            )

        tool = tools_from_registry(
            registry, ["corral.who"], context_factory=ctx_factory
        )[0]

        with acting_as(pc.PrincipalContext(handle="@zavier:tamu")):
            tool.invoke({})

        assert seen["handle"] == "@zavier:tamu"
