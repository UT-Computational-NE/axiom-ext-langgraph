# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""Identity from the skill context all the way to the model call.

The gap this closes was named before it was measured, and measuring it found
something worse than "not wired": it was wired to a type the platform silently
refuses.

``SkillContext.principal`` is a ``PrincipalContext``. ``get_current_actor``
returns what was bound only if it is a governance ``Principal``, and otherwise
falls through — but "something is bound" stops the fallback chain, so it raises
instead. So a ported graph bound its principal, the gateway read back nothing,
the model call went out unattributed, and the ``AXIOM_ACTOR`` fallback was
disabled for the duration. Two type names that look alike.
"""

from __future__ import annotations

import pytest

from axiom_ext_langgraph.actor import acting_as, current_actor


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


def _handle_seen_by_the_gateway() -> str | None:
    from axiom_ext_langgraph.chat_model import _current_principal_handle

    return _current_principal_handle()


class TestWhatTheGatewayActuallySees:
    def test_a_principal_context_reaches_the_model_call(self):
        """The shape every skill invocation actually carries."""
        _needs_principal_conversion()
        pc = pytest.importorskip("axiom.infra.principal", reason="needs the platform")

        with acting_as(pc.PrincipalContext(handle="@ben:ut")):
            assert _handle_seen_by_the_gateway() == "@ben:ut"

    def test_a_bare_handle_works_too(self):
        _needs_principal_conversion()
        with acting_as("@nima:netl"):
            assert _handle_seen_by_the_gateway() == "@nima:netl"

    def test_a_real_principal_passes_through_unchanged(self):
        governance = _needs_principal_conversion()
        principal = governance.principal_from_handle("@zavier:tamu")

        with acting_as(principal) as bound:
            assert bound is principal
            assert _handle_seen_by_the_gateway() == "@zavier:tamu"


class TestBindingNothingIsBetterThanBindingRubbish:
    def test_none_leaves_the_ambient_actor_alone(self):
        _needs_principal_conversion()
        with acting_as("@ambient:host"), acting_as(None):
            assert _handle_seen_by_the_gateway() == "@ambient:host"

    def test_an_object_with_no_handle_binds_nothing(self):
        _needs_principal_conversion()
        """Replacing a working actor with an unreadable one is a downgrade.

        The platform treats "something is bound" as "stop looking", so binding
        rubbish does not merely fail to attribute — it disables the fallback.
        """
        with acting_as("@ambient:host"), acting_as(object()):
            assert _handle_seen_by_the_gateway() == "@ambient:host"


class TestItDoesNotLeakPastTheBlock:
    def test_the_first_binding_in_a_process_is_cleared(self):
        _needs_principal_conversion()
        """It was not. ``acting_as`` restored only a non-None previous, so the
        first binding outlived its block and every later unattributed operation
        was recorded as that graph's principal."""
        assert current_actor() is None or _handle_seen_by_the_gateway() != "@leaky:one"

        with acting_as("@leaky:one"):
            assert _handle_seen_by_the_gateway() == "@leaky:one"

        assert _handle_seen_by_the_gateway() != "@leaky:one"

    def test_nesting_restores_the_outer_actor(self):
        _needs_principal_conversion()
        with acting_as("@outer:x"):
            with acting_as("@inner:y"):
                assert _handle_seen_by_the_gateway() == "@inner:y"
            assert _handle_seen_by_the_gateway() == "@outer:x"

    def test_a_raising_block_still_restores(self):
        _needs_principal_conversion()
        """A failed graph run must not leave someone else's principal bound."""
        with acting_as("@outer:x"):
            with pytest.raises(RuntimeError), acting_as("@inner:y"):
                raise RuntimeError("the graph failed")
            assert _handle_seen_by_the_gateway() == "@outer:x"


class TestThroughAPortedGraph:
    """The whole path: a capability invoked with a context, a model call inside."""

    def test_the_graphs_model_call_is_attributed_to_the_caller(self):
        _needs_principal_conversion()
        pc = pytest.importorskip("axiom.infra.principal", reason="needs the platform")
        from axiom.infra.skills import SkillContext, SkillRegistry

        from axiom_ext_langgraph.graphs import skill_from_graph

        seen: dict[str, str | None] = {}

        class GraphThatCallsAModel:
            def invoke(self, state, *a, **k):
                # Stands in for AxiomChatModel._generate, which resolves the
                # principal at exactly this moment and passes it to the gateway.
                seen["principal"] = _handle_seen_by_the_gateway()
                return {"answer": "done"}

        import logging

        ctx = SkillContext(
            registry=SkillRegistry(),
            state_dir=__import__("pathlib").Path("/tmp"),
            logger=logging.getLogger("t"),
            principal=pc.PrincipalContext(handle="@ben:ut"),
        )

        result = skill_from_graph(GraphThatCallsAModel())({"prompt": "hi"}, ctx)

        assert result.ok
        assert seen["principal"] == "@ben:ut", (
            "the model call inside the graph must carry the caller's identity"
        )

    def test_without_the_fix_it_would_have_been_unattributed(self):
        """Pins the failure, so a regression is loud rather than silent.

        Binding the raw PrincipalContext is what the shim used to do.
        """
        _needs_principal_conversion()
        pc = pytest.importorskip("axiom.infra.principal", reason="needs the platform")
        governance = _needs_principal_conversion()

        governance.set_current_actor(pc.PrincipalContext(handle="@ben:ut"))
        try:
            assert _handle_seen_by_the_gateway() is None, (
                "a PrincipalContext bound directly is invisible to the gateway; "
                "if this ever starts working, _as_bindable can be simplified"
            )
        finally:
            governance.set_current_actor(None)
