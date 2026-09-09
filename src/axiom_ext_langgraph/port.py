# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""``python -m axiom_ext_langgraph.port`` — port a LangGraph project in one command.

    python -m axiom_ext_langgraph.port path/to/langgraph.json --namespace moose

Reads the project's own manifest, writes the registration module, and prints
the two lines to add to ``pyproject.toml``. What it will not do is pretend the
whole port is mechanical: the parts that are a decision get printed as
decisions, with the reason, rather than silently defaulted.

Why this exists rather than a migration guide
---------------------------------------------

A guide asks somebody to hand-translate a manifest they already wrote. The two
manifests are the same declaration in different spellings — ``graphs`` maps a
name to ``"file.py:object"``, and ``register_entry`` maps a name to
``"module:function"`` — so the translation is a program's job.

What is genuinely a decision, and is surfaced as one
-----------------------------------------------------

**A ``checkpointer`` in the manifest.** It means the graph is stateful and
expects to resume. Ported, it should pause on the platform's ``ApprovalGate``,
which is durable and is the single record of what was allowed. Two checkpoint
stores fork that record, and a safety case cannot cite two. Flagged, never
auto-wired.

**Surfaces for a graph that writes.** The default gives a ported capability
every surface, which is right for the common case. A graph that writes wants
its surfaces narrowed the way an approval verb is CLI-only, and only its author
knows which it is.

**``auth``.** LangGraph's auth config and Axiom's principal are the same
concern reached two ways. Ported capabilities are already attributed through
the skill context, so the config usually becomes redundant rather than
translated, but confirming that is somebody's judgement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from axiom_ext_langgraph.graphs import (
    LangGraphManifestError,
    read_langgraph_manifest,
)

__all__ = ["build_registration_module", "main", "report"]

_MODULE_TEMPLATE = '''"""Axiom capability registration, generated from langgraph.json.

Each graph the project already declares is registered as a capability. Nothing
here re-describes a graph; the manifest stays the single declaration and this
module is regenerated from it.

    python -m axiom_ext_langgraph.port {manifest} --namespace {namespace}
"""

from pathlib import Path

from axiom_ext_langgraph.graphs import skills_from_langgraph_json

MANIFEST_NAME = "{manifest_name}"
NAMESPACE = "{namespace}"


def _find_manifest():
    """Walk up from this module to the manifest.

    Not ``__file__.parent / MANIFEST_NAME``: langgraph.json sits at the project
    root while this module sits inside the package, so the sibling form is
    wrong in exactly the layout every real project has. Walking up finds the
    nearest one, and in a monorepo the nearest one is this project's.
    """
    here = Path(__file__).resolve().parent
    for directory in (here, *here.parents):
        candidate = directory / MANIFEST_NAME
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"no {{MANIFEST_NAME}} found at or above {{here}}. "
        "Regenerate with python -m axiom_ext_langgraph.port."
    )


def register_all(registry):
    """Entry point for ``axiom.skills``. Returns the capability names."""
    return skills_from_langgraph_json(registry, _find_manifest(), namespace=NAMESPACE)
'''


def build_registration_module(manifest: Path, namespace: str) -> str:
    """The text of the module that registers this project's graphs."""
    return _MODULE_TEMPLATE.format(
        manifest=manifest.as_posix(),
        manifest_name=manifest.name,
        namespace=namespace,
    )


def report(manifest: Path, namespace: str, distribution: str) -> list[str]:
    """Everything a person needs to see, in the order they need to see it."""
    ports = read_langgraph_manifest(manifest)
    raw = json.loads(manifest.read_text(encoding="utf-8"))

    lines = [
        f"Porting {manifest} to Axiom capabilities under namespace {namespace!r}.",
        "",
        f"{len(ports)} graph(s) become capabilities:",
        "",
    ]
    for p in ports:
        lines.append(f"  {p.name:<20} {p.path}")
        lines.append(f"  {'':<20} -> {namespace}.{p.name}   (entry {p.entry})")
    lines += [
        "",
        "Each one gains a CLI verb, an MCP tool, an agent-facing function and a",
        "generated SKILL.md from that single registration (ADR-072), plus an audit",
        "record and the caller's identity. None of that needs code.",
        "",
        "Add to pyproject.toml:",
        "",
        '  [project.entry-points."axiom.portfolio_member"]',
        f"  {distribution} = \"{namespace}:__name__\"",
        "",
        '  [project.entry-points."axiom.skills"]',
        f"  {distribution} = \"{namespace}.axiom_capabilities:register_all\"",
        "",
        "Both are required. The portfolio one is an authority boundary rather than",
        "paperwork: loading an entry point means importing and calling your code, so",
        "a distribution that does not declare membership is skipped and logged.",
        "",
    ]

    decisions: list[str] = []
    if raw.get("checkpointer"):
        decisions.append(
            "  checkpointer: your manifest configures one, so these graphs expect to\n"
            "    resume. Ported, they should pause on the platform's ApprovalGate,\n"
            "    which is durable, survives the process, and is the single record of\n"
            "    what was allowed. Two checkpoint stores fork that record. Not wired\n"
            "    automatically, because which graphs pause is yours to say."
        )
    if raw.get("auth"):
        decisions.append(
            "  auth: ported capabilities are attributed through the skill context's\n"
            "    principal already, so this config is usually redundant rather than\n"
            "    translated. Worth confirming rather than assuming."
        )
    decisions.append(
        "  surfaces: the generated registration opts every capability into all four\n"
        "    surfaces, which is right for a graph that reads. A graph that WRITES\n"
        "    should have its surfaces narrowed the way an approval verb is CLI-only,\n"
        "    or an agent gains the ability to invoke it unattended."
    )

    lines += ["Decisions this tool will not make for you:", ""]
    lines += decisions
    lines += [
        "",
        "Then: axi ext lint && axi ext validate && axi ext doctor",
    ]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m axiom_ext_langgraph.port",
        description="Port a LangGraph project's graphs to Axiom capabilities.",
    )
    parser.add_argument("manifest", type=Path, help="path to langgraph.json")
    parser.add_argument(
        "--namespace",
        required=True,
        help="capability prefix, conventionally the extension's CLI noun",
    )
    parser.add_argument(
        "--distribution",
        default="",
        help="distribution name for the entry points (default: the namespace)",
    )
    parser.add_argument(
        "--write",
        type=Path,
        default=None,
        help="write the registration module here instead of printing it",
    )
    args = parser.parse_args(argv)

    try:
        lines = report(
            args.manifest, args.namespace, args.distribution or args.namespace
        )
    except LangGraphManifestError as exc:
        print(f"cannot port: {exc}", file=sys.stderr)
        return 1

    print("\n".join(lines))

    module = build_registration_module(args.manifest, args.namespace)
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(module, encoding="utf-8")
        print(f"\nWrote {args.write}")
    else:
        print("\n--- axiom_capabilities.py (pass --write to save) ---\n")
        print(module)
    return 0


if __name__ == "__main__":
    sys.exit(main())
