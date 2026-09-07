# axiom-ext-langgraph — contributor and agent guide

For humans and coding agents. `CLAUDE.md` symlinks here.

`README.md` says what this package is for. This file says how to work in it, and
what is knowingly missing.

---

## The rule this package exists to enforce

Hosting a foreign agent runtime is fine. Hosting a second *brain* is not. The
difference is whether there stays exactly one of each of four things:

| | | Status |
|---|---|---|
| 1 | **one LLM gateway** | done — `chat_model.py` |
| 2 | **one state and audit record** | **not done, on purpose** — see below |
| 3 | **one tool registry** | done — `tools.py` |
| 4 | **one identity** | partial — `actor.py`, with a platform-side gap |

If a change here would create a second of any of them, it is the wrong change,
however convenient.

---

## Fail closed, loudly

Three places deliberately raise rather than degrade. Do not soften them.

**A non-success gateway response raises.** The gateway returns placeholder text
with `success=False` when no provider is usable, which is right for a CLI that
can tell a human. Handed to a graph it becomes content the graph branches on —
a failure laundered into an answer. `AxiomGatewayUnavailable` stops the run.

**An unrecognised routing tier raises at construction.** Defaulting a mistyped
`export_controlled` to `any` would send controlled work to a public provider.

**A failed skill raises.** A graph must see a tool error it can branch on, not a
string that reads like a successful result.

Streaming also raises. LangChain's default would call `_generate` and yield one
chunk, which *looks* like streaming works. The gateway has `stream_with_tools`,
so this is a real gap, not an impossibility — implement it or leave it loud.

---

## Respect what the platform already declares

Two things were nearly reinvented here and should not be.

**Declared surfaces.** A `SkillSpec` names which projections it opts into —
`cli`, `mcp`, `agent_tool`, `skill_md`. That is the bounded-exposure guard
against tool explosion. Bulk projection honours it: a skill with undeclared
surfaces is *not* volunteered, because undeclared means the question was never
asked, not yes. Naming a skill explicitly projects it anyway — the caller opted
in on its behalf.

**Declared inputs.** `SkillSpec.inputs` gives a tool real named arguments. A
tool whose only parameter is an untyped blob makes the model guess.

---

## What is deliberately not here

**The durable checkpoint.** LangGraph's own checkpointer would be a second state
store, and two checkpoint stores fork the record of what an agent did — a safety
case cannot cite two records. The fix is a checkpointer backed by the Axiom
store.

It is not stubbed here because it is not a shim's job. The platform's
`ApprovalGate` already holds write actions for human confirmation but keeps them
in an in-memory dictionary, so a pause does not survive a restart, while the
session store beside it persists. A durable, resumable, time-bounded checkpoint
is a **platform primitive with an owner**, and it is the load-bearing mechanism
of a research thrust that depends on it. Building it here would take that
contribution rather than enable it.

The seam is declared in `axiom-extension.toml` as a surface touched in mode
`never`, so the gap is visible rather than assumed.

---

## A known gap on the platform's side

`acting_as` binds the actor for a graph run, and anything calling
`get_current_actor()` sees it. **`axiom.llm.gateway` does not call it.** There is
no principal parameter anywhere in the gateway, so the model call itself is not
attributed even while the surrounding graph is.

Binding here is therefore necessary and not yet sufficient. Single number four
is not fully satisfied, and no amount of work in this repo will satisfy it — the
gateway has to accept a principal. That is the platform's half of the identity
seam, and it should be fixed there rather than faked here.

Do not paper over this by, for example, stuffing a principal into `routing_tags`.
A tag is not an identity, and a fake attribution is worse than an absent one.

---

## Conventions

**Read the API, do not guess at it.** Every call here is written against source
that was actually opened, and the failure mode that discipline avoids is a
specific one: code written against an imagined interface calls a method that
does not exist, through a `getattr` default that swallows the miss, so it
returns empty results and raises nothing. It passes review and it passes CI. If
an interface you need does not exist, say so at runtime and return nothing —
never guess at a second shape, and never write a test that mocks an API you have
not opened.

**Tests take their dependencies as parameters.** `tool_from_skill` takes a
registry; `AxiomChatModel` takes a `gateway_factory`; `acting_as` resolves
`axiom.governance` lazily so a fake can be injected. The whole suite runs with
only `langchain-core` installed. A shim whose tests need the world stops being
run.

**Private attributes are assigned normally, never with `object.__setattr__`.**
Pydantic keeps them in `__pydantic_private__`; `object.__setattr__` puts a second
copy in `__dict__` that shadows it on read while later assignments go to the
other one. That cost an afternoon: `bind_tools` produced a model whose lazily
built gateway was written to one place and read from another, and `_get_gateway`
quietly returned `None`.

**Lint config is explicit.** Rules are selected in `pyproject.toml` so this repo
lints identically wherever it is checked out, rather than inheriting from
whatever directory it happens to sit under.

---

## Commands

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest src/axiom_ext_langgraph/tests/ -q
.venv/bin/python -m ruff check src/
```

---

_Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs. Apache-2.0 licensed._
