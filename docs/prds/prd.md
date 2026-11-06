# PRD — LangGraph toolkit and library

## Problem

LangGraph is a foreign agent runtime. Hosting it is fine. Hosting a second gateway, a second tool registry, a second identity, or a second audit record is not.

Callers need two things, and they are not the same thing:

1. A **toolkit** an agent or an operator can invoke: project registry skills into a graph, port a `langgraph.json` onto the registry, inspect a graph, and ask the approval gate about a host call.
2. A **library** that toolkit (and any host program) calls to do the work through the platform: one chat model, one actor binding, one projection, one graph port, one graph inspection, and one approval-gated host tool.

## Outcome

Installing this package is the integration step. The toolkit is declared in `axiom-extension.toml`. The library is the public Python API. Neither opens a provider, a registry, or an identity of its own.

## Non-goals

- A LangGraph checkpointer, including listing or resuming checkpoints. Pause and resume belong on the platform's approval gate. See `AGENTS.md`.
- A second MCP server. This extension projects the platform's registry; it does not publish another one.
- Gold federation features (behavioral attestation, quarantine). Those are runtime ceremonies this shim does not own.
