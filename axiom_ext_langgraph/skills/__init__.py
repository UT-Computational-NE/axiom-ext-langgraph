# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""Skill functions for the LangGraph toolkit.

The SkillSpec registered by :func:`bind` is the source of truth (ADR-063).
The matching ``[[extension.provides]]`` table in ``axiom-extension.toml`` is
its static projection. ``axi ext lint`` checks the two agree when the
platform is installed; this package does not invent a second projector.
"""

from __future__ import annotations

from typing import Any

from axiom_ext_langgraph.skills.impl import report

__all__ = ["bind", "bind_default", "report"]


def bind(registry: Any) -> None:
    """Register this extension's skills on ``registry``."""
    from axiom.infra.skills import SkillSpec

    registry.register_skill(
        SkillSpec(
            name="langgraph.report",
            fn=report,
            description=(
                "Read a langgraph.json and report the capability names it would "
                "become. Does not register them."
            ),
            inputs={"document": "str", "namespace": "str"},
            surfaces=("cli", "skill_md"),
            side_effects=False,
            idempotent=True,
            risk="read",
            requires=(),
        ),
        mutating=False,
    )


def bind_default() -> Any:
    """Bind into the process-wide default registry.

    ``axiom.infra.skills_emit`` imports an extension's ``skills`` package and
    calls ``bind_default`` when it is there. The name is the contract.
    """
    from axiom.infra.skills import default_registry

    registry = default_registry()
    bind(registry)
    return registry
