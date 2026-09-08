# axiom-ext-langgraph

Run LangChain and LangGraph through the Axiom gateway.

## Why this exists

Hosting a foreign agent runtime is fine. Hosting a second *brain* is not.

The platform already hosts Claude Code, Claude Desktop, Cursor, Codex and
Continue by translating their requests through one gateway, so every client
inherits the same tier routing, fail-closed export-control enforcement,
credential vault and audit. This package makes LangGraph the next client on that
list rather than an exception to it.

The difference between a client and a second brain is whether there stays
exactly **one** of each of four things:

| | | Here |
|---|---|---|
| 1 | one LLM gateway | `AxiomChatModel` |
| 2 | one state and audit record | **not implemented** — see below |
| 3 | one tool registry | `tools_from_registry` |
| 4 | one identity | `acting_as` |

If those are single, a LangGraph run is a rendering of the platform's behaviour.
If any is doubled, the split is real and the framework's name is irrelevant.

## Install

```bash
pip install axiom-ext-langgraph          # chat model + tools + actor binding
pip install "axiom-ext-langgraph[graph]" # adds langgraph itself
```

`langchain-core` is a hard dependency **here**, and deliberately not one of
Axiom's. That is the point: hosting LangGraph costs the platform nothing in its
own dependency graph.

## Use

```python
from langchain.agents import create_agent
from axiom_ext_langgraph import AxiomChatModel, acting_as, tools_from_registry

model = AxiomChatModel(routing_tier="export_controlled")
tools = tools_from_registry(registry, ["data.install", "telemetry.series"])

agent = create_agent(model, tools)

with acting_as(principal):
    result = agent.invoke({"messages": [("user", "what happened on Tuesday?")]})
```

`create_react_agent` from `langgraph.prebuilt` still works and is what most
tutorials show, but it moved to `langchain.agents.create_agent` in LangGraph 1.0
and goes away in 2.0. `examples/with_tools.py` prefers the new one and falls
back, because the `[graph]` extra installs `langgraph` without `langchain`.

Runnable versions of this live in `src/axiom_ext_langgraph/examples/`. Start
with `ask_once` (one model call, no tools) and then `with_tools`. Their tests
need no gateway, no network and no credentials, so you can iterate on a laptop.

Every model call in that graph goes through the gateway. There is no
configuration that lets it reach a provider directly.

### Pinning a tier for the whole process

```bash
export AXIOM_LANGGRAPH_ROUTING_TIER=export_controlled
```

Mirrors `AXIOM_BRIDGE_ROUTING_TIER` on the Anthropic/OpenAI ingress, so a
facility deployment configures both bridges the same way. An explicit
constructor argument wins over the environment, and an unrecognised tier is
**refused rather than defaulted** — silently treating a mistyped
`export_controlled` as `any` would send controlled work to a public provider.

## It fails closed

The gateway degrades gracefully for its own callers: when no provider is usable
it returns `success=False` alongside placeholder text, so a CLI can carry on and
tell the user. Handing that to a graph as ordinary content would launder a
failure into an answer the graph then branches on.

So a non-success response raises `AxiomGatewayUnavailable`. A graph halts rather
than reasoning about a sentence no model produced.

Streaming raises `NotImplementedError` for the same reason: LangChain's default
fallback would call `_generate` and yield a single chunk, which looks like
streaming works.

## What is deliberately not here

**The durable checkpoint.** LangGraph's own checkpointer is a second state
store, and two checkpoint stores fork the record of what an agent did — a safety
case cannot cite two records. The fix is a checkpointer backed by the Axiom
store, so pause/resume and the platform's receipt chain are the same rows.

That is not stubbed here because it is not a shim's job. The platform's approval
gate already holds write actions for confirmation but keeps them in an in-memory
dictionary, so a pause does not survive a restart. A durable, resumable,
time-bounded checkpoint is a **platform primitive with an owner**, and it is the
load-bearing mechanism of a research thrust that depends on it. Building it here
would take that contribution rather than enable it.

The seam is declared in `axiom-extension.toml` as a surface this shim touches
in mode `never`, so the gap is visible rather than assumed.

## Status

Early. The chat model, tool projection and actor binding are implemented and
tested. See `AGENTS.md` for the known gaps, including one on the platform's side
of the identity seam.

---

_Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs. Apache-2.0 licensed._
