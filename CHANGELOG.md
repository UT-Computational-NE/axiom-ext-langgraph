# Changelog

All notable changes to this extension are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this extension adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- LangGraph toolkit: `AxiomToolkit` over the one skill registry, the `project_skills` tool, the `axi langgraph port` command, and the read-only `langgraph.report` skill.
- Library module `axiom_ext_langgraph.library` for the gateway chat model, actor binding, tool projection, and graph port the toolkit is built on.
- AEOS 0.2 manifest at the repository root, compound layout, standard tests, and the ADR-099 surface table in `axiom_ext_langgraph/surfaces.py`.

### Changed

- Package layout moved out of `src/` so the extension directory matches the AEOS compound layout. Imports stay `axiom_ext_langgraph`.

## [0.1.0] - 2026-09-08

### Added

- `AxiomChatModel`, skill-to-tool projection, and `acting_as` actor binding.
- Graph port from `langgraph.json` to skill registrations.
- Tracing guard that strips LangChain-family telemetry configuration at import.
