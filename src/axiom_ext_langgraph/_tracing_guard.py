# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""Keep the count of external telemetry sinks at exactly zero.

The four singles in :mod:`axiom_ext_langgraph` say what there must be one of.
This module is about a thing there must be *none* of: a second, hosted place
where traces land. ``langsmith`` ships as a required transitive dependency of
``langchain-core`` and turns on by environment variable alone — no code change,
no import, no review. A retrieval trace carries the question text and the
retrieved chunks, and over a code's user guide or theory manual that is
export-scoped content leaving by a route nobody looked at.

The rule is the prefix, not an inventory: every ``LANGSMITH_*`` and
``LANGCHAIN_*`` variable is removed, then the master switches are pinned to
``"false"``, so a variable invented by a future release is covered the day it
ships. Turning external tracing on is still possible — by setting
``AXIOM_ALLOW_EXTERNAL_TRACING`` where it can be seen and reviewed, which is
the point.

Scope, stated precisely: the import-time call closes the *inherited-environment*
path. langsmith reads env at ``Client`` construction — lazily, on the first
trace — so a variable set after import would re-enable egress; invocation-path
call sites (``AxiomChatModel._generate``, and graph runners) re-enforce at the
last moment to close that window. What no environment guard can stop is code
that constructs a tracer or ``Client`` explicitly — that is a code-review
boundary, not an environment one.
"""

from __future__ import annotations

import logging
import os
from collections.abc import MutableMapping

logger = logging.getLogger(__name__)

ALLOW_ENV = "AXIOM_ALLOW_EXTERNAL_TRACING"

#: Every variable any LangChain-family reader treats as "tracing is on",
#: pinned explicitly off so an unset-vs-false ambiguity never arises.
MASTER_SWITCHES = (
    "LANGCHAIN_TRACING",
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_TRACING",
    "LANGSMITH_TRACING_V2",
)

_PREFIXES = ("LANGSMITH_", "LANGCHAIN_")
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def enforce_no_external_tracing(
    env: MutableMapping[str, str] | None = None,
) -> list[str] | None:
    """Strip LangChain-family configuration from ``env`` and pin tracing off.

    Returns the sorted names that were present and got removed (empty when the
    environment was already clean), or ``None`` when ``AXIOM_ALLOW_EXTERNAL_TRACING``
    is truthy and the environment was deliberately left alone.
    """
    env = os.environ if env is None else env

    if env.get(ALLOW_ENV, "").strip().lower() in _TRUTHY:
        logger.warning(
            "%s is set: external tracing configuration left untouched. "
            "Traces may leave this process for a hosted third party.",
            ALLOW_ENV,
        )
        return None

    neutralized = sorted(k for k in env if k.startswith(_PREFIXES))
    for key in neutralized:
        del env[key]
    for switch in MASTER_SWITCHES:
        env[switch] = "false"

    if neutralized:
        logger.warning(
            "neutralized external tracing configuration: %s",
            ", ".join(neutralized),
        )
    return neutralized


# Run at import. The package __init__ imports this module before any sibling
# that imports langchain, so the strip lands before any of that code can read
# the environment. Calling the function again later is harmless.
enforce_no_external_tracing()
