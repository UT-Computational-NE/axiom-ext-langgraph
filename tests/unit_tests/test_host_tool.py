# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""Host callables wrapped as LangChain tools, through the platform approval gate.

The gate is the real one. A read from its action registry runs. A write does
not run until that same gate says the action is approved, and a resume that
changes the parameters does not run either.
"""

from __future__ import annotations

import json

import pytest
from axiom.infra.orchestrator.actions import ActionStatus
from axiom.infra.orchestrator.approval import ApprovalGate

from axiom_ext_langgraph.commands.langgraph import cli


class _Calls:
    def __init__(self) -> None:
        self.params: list[dict] = []

    def __call__(self, **params: object) -> str:
        self.params.append(dict(params))
        return "ran"


def test_a_read_runs_and_is_recorded_completed():
    from axiom_ext_langgraph.host_tool import tool_from_host

    gate = ApprovalGate()
    calls = _Calls()
    tool = tool_from_host(
        calls,
        "lookup",
        gate=gate,
        action_name="query_docs",
        description="Read the docs.",
        inputs={"q": "str"},
    )

    assert tool.invoke({"q": "tuesday"}) == "ran"
    assert calls.params == [{"q": "tuesday"}]
    action = gate.all_actions()[-1]
    assert action.status == ActionStatus.COMPLETED
    assert action.category.value == "read"


def test_a_write_does_not_run_until_the_same_gate_approves_it():
    from axiom_ext_langgraph.host_tool import HostToolHeld, tool_from_host

    gate = ApprovalGate()
    calls = _Calls()
    tool = tool_from_host(
        calls,
        "doc_publish",
        gate=gate,
        description="Publish a document.",
        inputs={"source": "str"},
    )

    with pytest.raises(HostToolHeld, match="pending") as held:
        tool.invoke({"source": "docs/prds/prd_foo.md"})

    assert calls.params == []
    assert held.value.status == "pending"
    action = gate.get(held.value.action_id)
    assert action is not None
    assert action.status == ActionStatus.PENDING

    gate.approve(held.value.action_id, decided_by="@ben:ut")
    assert tool.invoke({"source": "docs/prds/prd_foo.md", "approval_id": held.value.action_id}) == "ran"
    assert calls.params == [{"source": "docs/prds/prd_foo.md"}]
    assert gate.get(held.value.action_id).status == ActionStatus.COMPLETED

    with pytest.raises(HostToolHeld):
        tool.invoke({"source": "docs/prds/prd_foo.md", "approval_id": held.value.action_id})
    assert len(calls.params) == 1


def test_a_rejected_write_never_runs():
    from axiom_ext_langgraph.host_tool import HostToolHeld, tool_from_host

    gate = ApprovalGate()
    calls = _Calls()
    tool = tool_from_host(calls, "doc_publish", gate=gate, inputs={"source": "str"})

    with pytest.raises(HostToolHeld) as held:
        tool.invoke({"source": "docs/prds/prd_foo.md"})
    gate.reject(held.value.action_id, "wrong revision", decided_by="@ben:ut")

    with pytest.raises(HostToolHeld, match="rejected"):
        tool.invoke({"source": "docs/prds/prd_foo.md", "approval_id": held.value.action_id})
    assert calls.params == []


def test_a_resume_that_changes_the_approved_params_does_not_run():
    from axiom_ext_langgraph.host_tool import HostToolHeld, tool_from_host

    gate = ApprovalGate()
    calls = _Calls()
    tool = tool_from_host(calls, "doc_publish", gate=gate, inputs={"source": "str"})

    with pytest.raises(HostToolHeld) as held:
        tool.invoke({"source": "docs/a.md"})
    gate.approve(held.value.action_id, decided_by="@ben:ut")

    with pytest.raises(HostToolHeld, match="mismatch"):
        tool.invoke({"source": "docs/b.md", "approval_id": held.value.action_id})
    assert calls.params == []
    assert gate.get(held.value.action_id).status == ActionStatus.APPROVED

    assert tool.invoke({"source": "docs/a.md", "approval_id": held.value.action_id}) == "ran"
    assert calls.params == [{"source": "docs/a.md"}]


def test_an_unknown_action_name_is_a_write():
    from axiom_ext_langgraph.host_tool import HostToolHeld, decide_host_call

    gate = ApprovalGate()
    decision = decide_host_call(gate, "host.custom_write", {"n": 1})

    assert decision.allowed is False
    assert decision.status == "pending"
    assert decision.category == "write"
    assert decision.as_dict()["name"] == "host.custom_write"
    with pytest.raises(HostToolHeld):
        from axiom_ext_langgraph.host_tool import tool_from_host

        tool_from_host(lambda **_params: "no", "host.custom_write", gate=gate).invoke({})


def test_a_missing_approval_does_not_run():
    from axiom_ext_langgraph.host_tool import HostToolHeld, tool_from_host

    calls = _Calls()
    tool = tool_from_host(calls, "doc_publish", gate=ApprovalGate(), inputs={"source": "str"})

    with pytest.raises(HostToolHeld, match="missing"):
        tool.invoke({"source": "docs/a.md", "approval_id": "does-not-exist"})
    assert calls.params == []


def test_a_failed_host_call_is_recorded_and_not_retried(monkeypatch):
    from axiom_ext_langgraph.host_tool import HostToolHeld, tool_from_host

    gate = ApprovalGate()

    def blow_up(**_params: object) -> None:
        raise RuntimeError("disk full")

    tool = tool_from_host(blow_up, "lookup", gate=gate, action_name="query_docs")
    with pytest.raises(RuntimeError, match="disk full"):
        tool.invoke({})
    action = gate.all_actions()[-1]
    assert action.status == ActionStatus.FAILED

    with pytest.raises(HostToolHeld):
        tool.invoke({"approval_id": action.action_id})


def test_gate_command_and_tool_share_decide_host_call(capsys):
    from axiom_ext_langgraph.host_tool import decide_host_call
    from axiom_ext_langgraph.tools.gate_host_call import GateHostCall

    gate = ApprovalGate()
    expected = decide_host_call(gate, "doc_publish", {"source": "docs/a.md"}).as_dict()
    # A second gate, so the tool's submit is its own action. Compare the
    # decision shape, not the action id, against a fresh submit below.
    tool_gate = ApprovalGate()
    GateHostCall.bind_gate(tool_gate)
    try:
        reported = GateHostCall.invoke({"name": "doc_publish", "params": {"source": "docs/a.md"}})
    finally:
        GateHostCall.bind_gate(None)

    assert reported["allowed"] is False
    assert reported["status"] == "pending"
    assert reported["category"] == "write"
    assert reported["name"] == expected["name"]
    assert tool_gate.get(reported["action_id"]).status == ActionStatus.PENDING

    command_gate = ApprovalGate()
    assert cli(["gate", "--name", "query_docs", "--params", json.dumps({"q": "x"})], gate=command_gate) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["allowed"] is True
    assert body["status"] == "approved"
    assert body["category"] == "read"

    write_gate = ApprovalGate()
    assert (
        cli(
            ["gate", "--name", "doc_publish", "--params", json.dumps({"source": "docs/a.md"})],
            gate=write_gate,
        )
        == 3
    )
    held = json.loads(capsys.readouterr().out)
    assert held["allowed"] is False
    write_gate.approve(held["action_id"], decided_by="@ben:ut")
    assert (
        cli(
            [
                "gate",
                "--name",
                "doc_publish",
                "--params",
                json.dumps({"source": "docs/a.md"}),
                "--approval-id",
                held["action_id"],
            ],
            gate=write_gate,
        )
        == 0
    )


def test_the_durable_gate_is_the_platform_file_store(tmp_path):
    from axiom_ext_langgraph.host_tool import decide_host_call, open_platform_gate

    path = tmp_path / "approvals.json"
    first = open_platform_gate(path)
    decision = decide_host_call(first, "doc_publish", {"source": "docs/a.md"})
    assert decision.allowed is False

    second = open_platform_gate(path)
    found = second.get(decision.action_id)
    assert found is not None
    assert found.status == ActionStatus.PENDING
    assert found.params == {"source": "docs/a.md"}
