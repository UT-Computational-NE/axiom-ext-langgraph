# ADR-001 — Toolkit, library, and where shim surfaces live

**Status:** Accepted
**Date:** 2026-10-23

## Context

AEOS names capability kinds (tool, cmd, skill, …). It does not name "toolkit" or "library". Axiom's own split, visible in `axiom.compute` ("this is the library layer; no CLI") and in ADR-099 (one shim, declared surfaces), is:

- A **toolkit** is the surface an agent or an operator invokes.
- A **library** is the importable API that surface is built on.

The AEOS manifest schema closes the root. `[[shim.surface]]` does not validate. ADR-099 leaves the declaration in a typed table when the manifest cannot hold it.

## Decision

This extension ships both:

- Toolkit: `AxiomToolkit`, `project_skills`, `axi langgraph port`, `langgraph.report`.
- Library: `axiom_ext_langgraph.library`, which is also the package's public API.

Shim surfaces move to `axiom_ext_langgraph.surfaces.SURFACES`. The checkpoint store remains mode `never`.

`aeos_version` is `0.2.0`, the version `axi ext init` stamps. AEOS 1.0 does not govern extension conformance while it is in draft.

## Consequences

Bronze layout and manifest checks apply at the repository root, with the package beside the manifest. The GitHub repository name stays hyphenated (`axiom-ext-langgraph`), which is the distribution name. `axi ext lint` compares the directory name to the manifest name with no hyphen folding, so lint the checkout through a path whose last component is `axiom_ext_langgraph`.
