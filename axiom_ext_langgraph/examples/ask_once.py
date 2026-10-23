# Copyright (c) 2026 The University of Texas at Austin
# SPDX-License-Identifier: Apache-2.0

"""The smallest graph that is still real: one model call, through the gateway.

Start here. There is exactly one thing to notice, and it is what you would
otherwise have written differently.

Upstream LangChain teaches you to construct a provider chat model::

    from langchain_anthropic import ChatAnthropic
    model = ChatAnthropic(model="claude-...", api_key=...)

Here you construct ``AxiomChatModel`` instead. It is a ``BaseChatModel``, so
every LangChain and LangGraph idiom you learn elsewhere works on it unchanged.
What differs is that it has no ``api_key`` argument and no endpoint: there is no
configuration that lets it reach a provider directly. Tier routing,
export-control enforcement, the credential vault and the audit record are not
things you opt into, because they are not things you could opt out of.
"""

from __future__ import annotations

from typing import Any


def ask_once(question: str, *, principal: str, routing_tier: str = "any", model=None) -> str:
    """Ask one question through the gateway and return the answer text.

    Args:
        question: What to ask.
        principal: Who this runs as, e.g. ``@nima:netl``. Required, and not a
            formality — an unattributed call is refused on the export-controlled
            tier, and an unattributed *record* is useless everywhere else.
        routing_tier: ``any``, ``public`` or ``export_controlled``. An
            unrecognised tier is refused rather than defaulted, because
            silently treating a mistyped ``export_controlled`` as ``any`` would
            send controlled work to a public provider.
        model: Injectable for tests. Production passes nothing.

    Raises:
        AxiomGatewayUnavailable: When no provider could answer. The gateway
            degrades politely for its own callers by returning placeholder text;
            handing that to a graph would launder a failure into an answer the
            graph then branches on, so the shim raises instead.
    """
    from axiom_ext_langgraph import AxiomChatModel, acting_as

    chat = model if model is not None else AxiomChatModel(routing_tier=routing_tier)

    with acting_as(principal):
        reply = chat.invoke([("user", question)])

    return _text_of(reply)


def _text_of(message: Any) -> str:
    content = getattr(message, "content", message)
    return content if isinstance(content, str) else str(content)
