# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""Surfaces this shim touches, per ADR-099 D1.

The declaration used to live in ``axiom-extension.toml`` as ``[[shim.surface]]``.
The AEOS manifest schema's root is closed (``additionalProperties: false``),
so a shim table is not a valid manifest key. ADR-099 leaves the home of the
declaration open and names typed tables, the same shape absorb specs use, as
the alternative. This module is that table. Adding a surface is a reviewable
change here, not a quiet edit in a call site.

Modes, from ADR-099:

- ``read`` — absorb, never write
- ``write-extension-point`` — a documented user-config surface
- ``restore-only`` — write only into paths that are absent
- ``never`` — vendor-owned state this shim does not touch
"""

from __future__ import annotations

from typing import NamedTuple

__all__ = ["SURFACES", "Surface"]


class Surface(NamedTuple):
    name: str
    mode: str
    note: str


SURFACES: tuple[Surface, ...] = (
    Surface(
        name="axiom.llm.gateway",
        mode="read",
        note=(
            "Every model call is translated through the gateway, so tier routing, "
            "fail-closed export-control enforcement, the vault and audit apply. "
            "Not a passthrough."
        ),
    ),
    Surface(
        name="axiom.infra.skills.SkillRegistry",
        mode="write-extension-point",
        note=(
            "Projection reads skills other extensions already registered and "
            "does not copy them. The one skill this extension writes, "
            "langgraph.report, is registered through the platform's bind_default "
            "hook, which is the documented registration point. It is a read of "
            "a manifest, not a second registry."
        ),
    ),
    Surface(
        name="axiom.governance actor context",
        mode="write-extension-point",
        note=(
            "The shim binds the current actor for the duration of a graph run "
            "and restores the previous value after."
        ),
    ),
    Surface(
        name="langgraph checkpoint store",
        mode="never",
        note=(
            "Not implemented here on purpose. A durable, resumable checkpoint "
            "backed by the Axiom store is a platform primitive with an owner; "
            "see AGENTS.md."
        ),
    ),
)
