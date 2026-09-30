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


def _as_bindable(principal: Any) -> Any | None:
    """Turn whatever the caller has into something the platform will read back.

    ``set_current_actor`` accepts any object and ``get_current_actor`` returns
    it only if it is a governance ``Principal``. Everything else is ignored —
    and not quietly ignored, but ignored in a way that then *raises*
    ``AuthnUnavailable`` for the rest of the block, because the fallback chain
    treats "something is bound" as "stop looking".

    A skill context carries a ``PrincipalContext``, which has a handle and is
    not a ``Principal``. Binding it directly, which is what this used to do,
    lost the attribution AND defeated the ``AXIOM_ACTOR`` fallback for anything
    inside the run. So a graph's model call went out unattributed, and the
    reason was two type names that look alike.

    Anything with a ``.handle`` is converted through the platform's own
    ``principal_from_handle`` rather than a local reimplementation.
    """
    if principal is None:
        return None

    try:
        from axiom.vega.identity.principal import Principal

        if isinstance(principal, Principal):
            return principal
    except Exception:
        pass

    handle = getattr(principal, "handle", None)
    if handle is None and isinstance(principal, str):
        handle = principal
    if not handle:
        return None

    try:
        from axiom.governance import principal_from_handle

        return principal_from_handle(str(handle))
    except Exception:
        # Best effort: bind what we were handed. This path exists for the fake
        # governance module the tests inject, which has the two actor functions
        # and nothing else, and for a platform predating principal_from_handle.
        # Binding the original is what this always did; it is only wrong when
        # the real platform is present, and there the branch above has already
        # returned.
        return principal


@contextmanager
def acting_as(principal: Any) -> Iterator[Any]:
    """Bind ``principal`` as the current actor for the enclosing block.

    The previous actor is restored on exit, including when the block raises, so
    a failed graph run cannot leave someone else's principal bound.

    Accepts a governance ``Principal``, a ``PrincipalContext``, or a bare
    handle string. Anything unusable binds nothing at all rather than binding
    something the platform will refuse to read: leaving the ambient actor in
    place is strictly better than replacing it with a value that makes every
    later lookup raise.

    Args:
        principal: a Principal, anything with a ``.handle``, or a handle.

    Yields:
        What was actually bound, or ``None`` when nothing was.
    """
    from axiom.governance import get_current_actor, set_current_actor

    bindable = _as_bindable(principal)
    if bindable is None:
        yield None
        return

    try:
        previous = get_current_actor()
    except Exception:
        # No actor bound yet, or resolution refused without dev_mode. Either
        # way there is nothing to restore, and narrowing this would couple the
        # shim to whichever exception the platform currently raises.
        previous = None

    set_current_actor(bindable)
    try:
        yield bindable
    finally:
        # Restore even when there was nothing before, by clearing. The old
        # version only restored a non-None previous, so the FIRST binding in a
        # process outlived its block and stayed current for everything after —
        # the next unattributed operation would be recorded as this graph's
        # principal. Setting None is safe: the platform ignores a non-Principal
        # and falls through to its normal resolution.
        set_current_actor(previous)


def current_actor() -> Any | None:
    """The currently bound actor, or ``None`` when there is not one."""
    from axiom.governance import get_current_actor

    try:
        return get_current_actor()
    except Exception:
        return None
