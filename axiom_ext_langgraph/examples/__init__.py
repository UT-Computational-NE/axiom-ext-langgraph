# Copyright (c) 2026 The University of Texas at Austin
# SPDX-License-Identifier: Apache-2.0

"""Runnable LangGraph agents that go through the platform. See ``RECIPE.md``.

If you are here to learn LangGraph, learn it on these. Everything you would
learn from the upstream tutorials still applies — a graph is a graph, a tool is
a tool, `create_react_agent` is the same function — and the only difference is
where the model and the tools come from.

That difference is the whole point, and it costs you three lines:

    model = AxiomChatModel()                       instead of ChatAnthropic(...)
    tools = tools_from_registry(registry, [...])   instead of hand-declared @tool
    with acting_as(principal):                     around the invoke

Learning LangGraph the upstream way and then porting is two jobs. Learning it
this way is one, and the graph you end up with is one the platform can route,
audit, and attribute — which is what makes it a thing that ships rather than a
thing that gets demoed.

The examples are ordered by what they teach:

``ask_once``      the smallest possible graph. One model call, no tools.
``with_tools``    a ReAct agent over registry tools, which is where the
                  difference from the tutorials actually shows.

Both are plain functions you can import and call. Their tests use a fake gateway
and need no credentials, no network and no VPN, which is the property that lets
you iterate on a laptop.
"""
