# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""Bind a principal for the duration of a graph run.

An agent action with no bound principal cannot be authorised, attributed, or
placed on an authority ladder. A graph is exactly the kind of long, multi-step
run where "who asked for this" is easiest to lose and most expensive to lack.

:func:`acting_as` binds the actor for the run and restores the previous value
afterwards, so a graph invoked from inside another actor's session cannot leak
its principal into the caller's context.

**A known gap, stated plainly.** The actor context is ambient — anything that
calls ``get_current_actor()`` sees it. ``axiom.llm.gateway`` does **not** call
it: there is no principal parameter anywhere in the gateway, so the model call
itself is not attributed even while the surrounding graph is. Binding here is
therefore necessary and not yet sufficient. Closing it is a platform change on
the gateway's side of this seam, not something a shim can fake. See AGENTS.md.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

__all__ = ["acting_as", "current_actor"]


@contextmanager
def acting_as(principal: Any) -> Iterator[Any]:
    """Bind ``principal`` as the current actor for the enclosing block.

    The previous actor is restored on exit, including when the block raises, so
    a failed graph run cannot leave someone else's principal bound.

    Args:
        principal: an ``axiom.governance`` Principal.

    Yields:
        The bound principal.
    """
    from axiom.governance import get_current_actor, set_current_actor

    try:
        previous = get_current_actor()
    except Exception:
        # No actor bound yet, or resolution refused without dev_mode. Either
        # way there is nothing to restore, and narrowing this would couple the
        # shim to whichever exception the platform currently raises.
        previous = None

    set_current_actor(principal)
    try:
        yield principal
    finally:
        if previous is not None:
            set_current_actor(previous)


def current_actor() -> Any | None:
    """The currently bound actor, or ``None`` when there is not one."""
    from axiom.governance import get_current_actor

    try:
        return get_current_actor()
    except Exception:
        return None
