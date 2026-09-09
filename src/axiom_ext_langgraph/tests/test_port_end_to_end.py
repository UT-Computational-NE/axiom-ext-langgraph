# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""The whole claim, end to end: a graph pauses on the platform and resumes.

Everything else in this package tests a piece. This tests the sentence the
package exists to make true:

    a LangGraph flow becomes an Axiom capability, can be held for a human by a
    site rule, survives the process that proposed it, and runs when a person
    says so — with no LangGraph checkpointer anywhere.

That last clause is the point. LangGraph's own checkpointer would give a graph
pause and resume out of the box, and if the platform could not match it,
"port to Axiom" would mean "give up resumability". It matches it, and the pause
lives in the one record of what an agent was allowed to do rather than a second
one beside it.

Written because the port had been tested in pieces and never as a whole: the
adapter against a real compiled graph, the manifest reader against a real
project, but never a real graph running through the platform's own dispatch.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys

import pytest

pytestmark = pytest.mark.graph


def _needs(module: str, why: str):
    return pytest.importorskip(module, reason=why)


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A real LangGraph project on disk, ported the way a person would port it."""
    _needs("langgraph.graph", "needs a real LangGraph")
    monkeypatch.setenv("AXI_STATE_DIR", str(tmp_path / "state"))

    root = tmp_path / "proj"
    pkg = root / "src" / "demo_agent"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "graph.py").write_text(
        '''from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class State(TypedDict, total=False):
    messages: list
    mesh: str
    deck: str
    steps: list


def _read(state):
    text = state["messages"][-1][1] if state.get("messages") else ""
    return {"mesh": "core.e" if "core" in text else "default.e", "steps": ["read"]}


def _build(state):
    return {
        "deck": f"[Mesh]\\n  file = {state['mesh']}\\n",
        "steps": [*state.get("steps", []), "build"],
    }


builder = StateGraph(State)
builder.add_node("read", _read)
builder.add_node("build", _build)
builder.add_edge(START, "read")
builder.add_edge("read", "build")
builder.add_edge("build", END)

graph = builder.compile()
'''
    )
    (root / "langgraph.json").write_text(
        json.dumps(
            {
                "dependencies": ["."],
                "graphs": {
                    "deck": {
                        "path": "./src/demo_agent/graph.py:graph",
                        "description": "Builds a reactor input deck from a prompt.",
                    }
                },
            }
        )
    )

    from axiom_ext_langgraph.port import build_registration_module

    (pkg / "axiom_capabilities.py").write_text(
        build_registration_module(root / "langgraph.json", "deckbuilder")
    )

    monkeypatch.syspath_prepend(str(root / "src"))
    return root


def _registry(project):
    from axiom.infra.skills import SkillRegistry

    path = project / "src" / "demo_agent" / "axiom_capabilities.py"
    spec = importlib.util.spec_from_file_location("generated_caps", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["generated_caps"] = module
    spec.loader.exec_module(module)

    registry = SkillRegistry()
    names = module.register_all(registry)
    return registry, names


def _ctx(registry, actor):
    from axiom.infra.paths import get_user_state_dir
    from axiom.infra.skills import SkillContext

    return SkillContext(
        registry=registry,
        state_dir=get_user_state_dir(),
        logger=logging.getLogger("e2e"),
        actor=actor,
    )


class TestAGraphBecomesACapability:
    def test_the_port_registers_it_with_every_surface(self, project):
        registry, names = _registry(project)

        assert names == ["deckbuilder.deck"]
        spec = registry.spec("deckbuilder.deck")
        assert set(spec.surfaces) == {"cli", "mcp", "agent_tool", "skill_md"}
        assert spec.description == "Builds a reactor input deck from a prompt."

    def test_it_runs_through_the_platform_dispatch_not_a_side_door(self, project):
        """``invoke_capability`` is the chokepoint: hooks, GUARD, audit."""
        _needs("axiom.infra.skill_dispatch", "needs the platform dispatch chokepoint")
        from axiom.infra.skill_dispatch import CLI_SURFACE, invoke_capability

        registry, _ = _registry(project)
        result = invoke_capability(
            registry,
            "deckbuilder.deck",
            {"prompt": "build me a core deck"},
            _ctx(registry, "@ben:ut"),
            surface=CLI_SURFACE,
        )

        assert result.ok, result.errors
        assert result.value["steps"] == ["read", "build"], "both nodes ran"
        assert "file = core.e" in result.value["deck"]


class TestItPausesOnThePlatformAndResumes:
    """The claim that decides whether porting costs anything."""

    def test_held_by_a_site_rule_then_run_when_a_human_says_so(
        self, project, monkeypatch
    ):
        _needs(
            "axiom.infra.orchestrator.runner",
            "needs the durable approval gate and runner (axiom with the "
            "orchestrator runner; skips until that release lands)",
        )
        import axiom.infra.tool_gateway as tool_gateway
        from axiom.infra.hooks import ApprovalRequired
        from axiom.infra.orchestrator.approval import ApprovalGate
        from axiom.infra.orchestrator.approval_store import (
            FileActionStore,
            default_approval_path,
        )
        from axiom.infra.orchestrator.runner import ActionRunner
        from axiom.infra.skill_dispatch import MCP_SURFACE, invoke_capability

        registry, _ = _registry(project)

        # A site rule: generated decks need a signoff. The agent is unattended.
        monkeypatch.setattr(
            tool_gateway,
            "dispatch_tool",
            lambda *a, **k: (_ for _ in ()).throw(
                ApprovalRequired("decks need a human signoff", hook_source="site")
            ),
        )
        refused = invoke_capability(
            registry,
            "deckbuilder.deck",
            {"prompt": "build me a core deck"},
            _ctx(registry, "@agent:autosam"),
            surface=MCP_SURFACE,
        )

        assert not refused.ok, "it must not have run"
        held = refused.value["held_action_id"]

        # A person, elsewhere, later.
        gate = ApprovalGate(FileActionStore(default_approval_path()))
        assert [a.action_id for a in gate.pending()] == [held]
        gate.approve(held, decided_by="@ben:ut")

        # A worker, elsewhere again. The hook is no longer demanding.
        monkeypatch.undo()
        monkeypatch.setenv("AXI_STATE_DIR", str(default_approval_path().parents[1]))
        report = ActionRunner(gate, registry, _ctx(registry, "@worker:local")).run_approved()

        assert report.completed == [held]
        done = gate.get(held)
        assert done.decided_by == "@ben:ut", "the record names the person"
        assert "file = core.e" in done.result["value"]["deck"], "the graph really ran"

    def test_a_rejected_graph_never_runs(self, project, monkeypatch):
        _needs("axiom.infra.orchestrator.runner", "needs the durable approval gate")
        import axiom.infra.tool_gateway as tool_gateway
        from axiom.infra.hooks import ApprovalRequired
        from axiom.infra.orchestrator.approval import ApprovalGate
        from axiom.infra.orchestrator.approval_store import (
            FileActionStore,
            default_approval_path,
        )
        from axiom.infra.orchestrator.runner import ActionRunner
        from axiom.infra.skill_dispatch import MCP_SURFACE, invoke_capability

        registry, _ = _registry(project)
        monkeypatch.setattr(
            tool_gateway,
            "dispatch_tool",
            lambda *a, **k: (_ for _ in ()).throw(
                ApprovalRequired("decks need a signoff", hook_source="site")
            ),
        )
        held = invoke_capability(
            registry,
            "deckbuilder.deck",
            {"prompt": "core"},
            _ctx(registry, "@agent:autosam"),
            surface=MCP_SURFACE,
        ).value["held_action_id"]

        gate = ApprovalGate(FileActionStore(default_approval_path()))
        gate.reject(held, "wrong mesh revision", decided_by="@ben:ut")
        monkeypatch.undo()

        assert ActionRunner(gate, registry, _ctx(registry, "@w:l")).run_approved().completed == []
        assert gate.get(held).result is None, "the graph must never have produced a deck"
