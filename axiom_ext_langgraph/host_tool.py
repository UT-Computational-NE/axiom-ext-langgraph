# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""Wrap a host callable as a LangChain tool that obeys the approval gate.

The gate is the platform's :class:`axiom.infra.orchestrator.approval.ApprovalGate`,
passed in by the caller. This module does not construct a second policy. It
submits an action with :func:`axiom.infra.orchestrator.actions.create_action`,
which classifies a known read as ``read`` and everything else as ``write``,
and then it trusts the action ``submit`` returns.

A returned status of ``approved`` is the only status that runs the callable.
``pending``, ``rejected``, ``completed``, ``failed``, and a missing record do
not. A resume must repeat the parameters that were submitted; a different
payload is a mismatch and does not run, and it does not consume the approval.

The command and the ``gate_host_call`` tool call :func:`decide_host_call` and
stop there. They record the ask. They do not run a callable. The LangChain
wrapper is what runs, and only after the same gate has approved that ask.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import Field, create_model

from axiom_ext_langgraph.tools.projection import surface_name

__all__ = [
    "HostDecision",
    "HostToolHeld",
    "decide_host_call",
    "open_platform_gate",
    "tool_from_host",
]

_APPROVED = "approved"


class HostToolHeld(RuntimeError):
    """The approval gate did not allow the call. The host callable did not run."""

    def __init__(self, name: str, action_id: str, status: str) -> None:
        self.name = name
        self.action_id = action_id
        self.status = status
        super().__init__(
            f"{name} is {status} under approval {action_id}; the host callable did not run"
        )


@dataclass(frozen=True)
class HostDecision:
    """What the gate said about one host call.

    ``action`` is the object the gate returned. Callers that only need to
    report the decision use :meth:`as_dict`, which leaves that object out.
    """

    allowed: bool
    action_id: str
    status: str
    category: str
    name: str
    action: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "action_id": self.action_id,
            "status": self.status,
            "category": self.category,
            "name": self.name,
        }


def _enum_value(obj: Any, *, what: str) -> str:
    if obj is None:
        raise TypeError(f"approval action has no {what}")
    value = getattr(obj, "value", obj)
    if not isinstance(value, str) or not value:
        raise TypeError(f"approval action {what} is not a string")
    return value


def _platform_create_action(name: str, params: dict[str, Any]) -> Any:
    from axiom.infra.orchestrator.actions import create_action

    return create_action(name, params)


def open_platform_gate(path: Path | None = None) -> Any:
    """The platform's file-backed approval gate.

    ``path`` defaults to :func:`axiom.infra.orchestrator.approval_store.default_approval_path`.
    The command uses that default so a write submitted from the CLI is the
    same queue ``axi approve`` reads. Tests pass a path of their own.
    """
    from axiom.infra.orchestrator.approval import ApprovalGate
    from axiom.infra.orchestrator.approval_store import FileActionStore, default_approval_path

    return ApprovalGate(FileActionStore(path if path is not None else default_approval_path()))


def decide_host_call(
    gate: Any,
    name: str,
    params: dict[str, Any],
    *,
    approval_id: str | None = None,
    action_factory: Callable[[str, dict[str, Any]], Any] | None = None,
) -> HostDecision:
    """Submit ``name`` to ``gate``, or look up an approval already given.

    The returned decision is the whole story. ``allowed`` is true only when
    the action the gate hands back has status ``approved`` and, on a resume,
    the parameters match the ones that were approved.
    """
    if not isinstance(name, str) or not name.strip():
        raise TypeError("name must be a non-empty string")
    if not isinstance(params, dict):
        raise TypeError("params must be an object")
    if approval_id is not None and (not isinstance(approval_id, str) or not approval_id):
        raise TypeError("approval_id must be a non-empty string")
    if gate is None or not callable(getattr(gate, "submit", None)) or not callable(getattr(gate, "get", None)):
        raise TypeError("gate must be an approval gate with submit() and get()")

    if approval_id is not None:
        action = gate.get(approval_id)
        if action is None:
            return HostDecision(False, approval_id, "missing", "", name, None)
        stored = getattr(action, "params", None)
        category = _enum_value(getattr(action, "category", None), what="category")
        if stored != params:
            return HostDecision(False, approval_id, "mismatch", category, name, action)
        status = _enum_value(getattr(action, "status", None), what="status")
        return HostDecision(status == _APPROVED, approval_id, status, category, name, action)

    factory = action_factory or _platform_create_action
    proposed = factory(name, params)
    if proposed is None:
        raise TypeError("action factory returned nothing")
    settled = gate.submit(proposed)
    if settled is None:
        raise TypeError("approval gate submit() returned nothing; the returned action is the record")
    action_id = getattr(settled, "action_id", None)
    if not isinstance(action_id, str) or not action_id:
        raise TypeError("approval action has no action_id")
    status = _enum_value(getattr(settled, "status", None), what="status")
    category = _enum_value(getattr(settled, "category", None), what="category")
    return HostDecision(status == _APPROVED, action_id, status, category, name, settled)


def _jsonish(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        return all(_jsonish(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _jsonish(item) for key, item in value.items())
    return False


def _record(gate: Any, decision: HostDecision, *, value: Any = None, error: str | None = None) -> None:
    action = decision.action
    if action is None:
        raise TypeError("cannot record a host call with no action")
    if not callable(getattr(gate, "record", None)):
        raise TypeError("approval gate has no record(); refusing to drop the outcome")
    if error is not None:
        if not callable(getattr(action, "fail", None)):
            raise TypeError("approval action has no fail()")
        action.fail(error)
    else:
        if not callable(getattr(action, "complete", None)):
            raise TypeError("approval action has no complete()")
        stored = value if _jsonish(value) else str(value)
        action.complete({"value": stored})
    gate.record(action)


def _args_model(name: str, inputs: dict[str, str] | None) -> type:
    fields: dict[str, Any] = {
        arg: (str, Field(description=shape or arg)) for arg, shape in (inputs or {}).items()
    }
    fields["approval_id"] = (
        str | None,
        Field(
            default=None,
            description="Resume a call this gate has already approved. Omit to submit a new action.",
        ),
    )
    return create_model(f"{surface_name(name)}_Args", **fields)


def tool_from_host(
    func: Callable[..., Any],
    name: str,
    *,
    gate: Any,
    description: str = "",
    inputs: dict[str, str] | None = None,
    action_name: str | None = None,
    action_factory: Callable[[str, dict[str, Any]], Any] | None = None,
) -> StructuredTool:
    """Project one host callable as a LangChain tool gated by ``gate``.

    ``action_name`` is what the gate classifies. It defaults to ``name``. A
    name the platform's action registry does not list is a write, which is
    the registry's own default, so an unrecognised host tool waits.

    The callable runs only when :func:`decide_host_call` allows it. Afterwards
    the action is completed, or failed if the callable raises, and recorded
    back through the same gate. A second call with that approval id does not
    run again.
    """
    if not callable(func):
        raise TypeError("func must be callable")
    if not isinstance(name, str) or not name.strip():
        raise TypeError("name must be a non-empty string")
    classified = action_name or name

    def _invoke(**params: Any) -> Any:
        approval_id = params.pop("approval_id", None)
        decision = decide_host_call(
            gate,
            classified,
            params,
            approval_id=approval_id,
            action_factory=action_factory,
        )
        if not decision.allowed:
            raise HostToolHeld(name, decision.action_id, decision.status)
        try:
            value = func(**params)
        except Exception as exc:
            _record(gate, decision, error=str(exc))
            raise
        _record(gate, decision, value=value)
        return value

    return StructuredTool.from_function(
        func=_invoke,
        name=surface_name(name),
        description=description or f"Host tool {name}",
        args_schema=_args_model(name, inputs),
    )
