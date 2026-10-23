# Changelog

All notable changes to this extension are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this extension adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- LangGraph toolkit: `AxiomToolkit` over the one skill registry, the `project_skills` tool, the `axi langgraph port` command, and the read-only `langgraph.report` skill.
- Graph inspection shared by the library, the `axi langgraph` verbs (`inspect`, `validate`, `mermaid`, `dry-run`), and the tools `inspect_graph`, `validate_graph`, `graph_mermaid`, and `dry_run_graph`.
- `tool_from_host`, which wraps a host callable as a LangChain tool and runs it only after the caller's `ApprovalGate` returns approved. `axi langgraph gate` and the `gate_host_call` tool ask that same gate and do not run the callable.
- Library module `axiom_ext_langgraph.library` for the gateway chat model, actor binding, tool projection, graph port, graph inspection, and the gated host tool.
- AEOS 0.1.0 manifest at the repository root, compound layout, standard tests, and the ADR-099 surface table in `axiom_ext_langgraph/surfaces.py`.

### Changed

- `aeos_version` is `0.1.0`. Chain level stays A. AEOS 0.2 remains a draft amendment this extension does not require.
- Package layout moved out of `src/` so the extension directory matches the AEOS compound layout. Imports stay `axiom_ext_langgraph`.

## [0.1.0] - 2026-09-08

### Added

- `AxiomChatModel`, skill-to-tool projection, and `acting_as` actor binding.
- Graph port from `langgraph.json` to skill registrations.
- Tracing guard that strips LangChain-family telemetry configuration at import.
