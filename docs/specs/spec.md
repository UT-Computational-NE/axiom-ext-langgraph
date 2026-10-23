# Spec — LangGraph extension

Parent: AEOS 0.1.0 (`aeos_version = "0.1.0"`). Chain level A is the default named in §4.11.1 and does not require the 0.2 draft. AEOS 1.0 is a draft and is not the governing extension spec.

## Toolkit

| Surface | Kind | Entry | What it does |
|---|---|---|---|
| `AxiomToolkit` | Python class, LangChain `BaseToolkit` | `axiom_ext_langgraph.toolkit:AxiomToolkit` | `get_tools()` projects the one skill registry |
| `project_skills` | `tool` | `axiom_ext_langgraph.tools.project_skills:ProjectSkills` | descriptors in, surface-safe tool specs out; no model call |
| `langgraph` | `cmd` | `axiom_ext_langgraph.commands.langgraph:cli` | `port`, plus `inspect`, `validate`, `mermaid`, `dry-run`, and `gate` |
| `inspect_graph`, `validate_graph`, `graph_mermaid`, `dry_run_graph` | `tool` | `axiom_ext_langgraph.tools` | the same graph report the command and the library return |
| `gate_host_call` | `tool` | `axiom_ext_langgraph.tools.gate_host_call:GateHostCall` | ask the caller's approval gate; do not run the call |
| `langgraph.report` | `skill` | `axiom_ext_langgraph.skills.impl:report` | read a `langgraph.json` text; do not register |

`project_skills` and `AxiomToolkit` share `surface_name` (`data.install` becomes `data_install`). Dots are not valid function-calling names. This spelling is local to the LangChain surface and is the one the existing tests lock. It is not a second capability projector for CLI or MCP names; those stay on the platform's `__` round-trip.

## Library

`axiom_ext_langgraph.library` re-exports, lazily:

- `AxiomChatModel`, `AxiomGatewayUnavailable` — the gateway
- `acting_as`, `current_actor` — identity
- `tool_from_skill`, `tools_from_registry`, `surface_name` — projection
- `GraphPort`, `read_langgraph_manifest`, `skill_from_graph`, `skills_from_langgraph_json` — the graph port
- `inspect_graph`, `validate_graph`, `to_mermaid`, `dry_run`, `shape_from_document`, `shape_from_compiled` — graph inspection
- `tool_from_host`, `decide_host_call`, `HostToolHeld` — a host callable through the caller's approval gate

The package root exports the same symbols plus `AxiomToolkit` and `ProjectSkills`.

## Surfaces

ADR-099 surface modes are `axiom_ext_langgraph.surfaces.SURFACES`. The manifest schema forbids a `shim` key. The checkpoint store stays mode `never`.

## Chain level

Claimed level is A. `langgraph.report` declares `risk = "read"` and `requires = []`. Level B verification runs in the platform's chain linter; this package does not reimplement it.
