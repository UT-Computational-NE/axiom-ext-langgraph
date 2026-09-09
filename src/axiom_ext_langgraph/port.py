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

__all__ = [
    "build_registration_module",
    "has_missing_targets",
    "install_into_extension",
    "main",
    "report",
]

_INIT_MARKER = "# axiom_ext_langgraph: generated graph capabilities"

_MODULE_TEMPLATE = '''"""Axiom capability registration, generated from langgraph.json.

Each graph the project already declares is registered as a capability. Nothing
here re-describes a graph; the manifest stays the single declaration and this
module is regenerated from it.

    python -m axiom_ext_langgraph.port {manifest} --namespace {namespace}

Drop this into your extension's ``skills`` package. The platform imports that
package and calls ``bind_default()``; that is the hook, and it is why the
function below has that name rather than a nicer one.
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
    """Register every declared graph into ``registry``. Returns the names."""
    return skills_from_langgraph_json(registry, _find_manifest(), namespace=NAMESPACE)


def bind_default():
    """What the platform calls. Binds into the process-wide default registry.

    The name is the contract: ``axiom.infra.skills_emit`` imports an
    extension's ``skills`` package and calls ``bind_default`` if it is there.
    Rename this and the capabilities register nowhere, silently.
    """
    from axiom.infra.skills import default_registry

    registry = default_registry()
    register_all(registry)
    return registry
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

    root = manifest.parent
    missing = [p for p in ports if p.missing_from(root)]

    lines = [
        f"Porting {manifest} to Axiom capabilities under namespace {namespace!r}.",
        "",
        f"{len(ports)} graph(s) become capabilities:",
        "",
    ]
    for p in ports:
        flag = "  MISSING" if p in missing else ""
        lines.append(f"  {p.name:<20} {p.path}{flag}")
        lines.append(f"  {'':<20} -> {namespace}.{p.name}   (entry {p.entry})")

    if missing:
        lines += [
            "",
            "STOP. These declarations point at files that are not there:",
            "",
        ]
        for p in missing:
            lines.append(f"  {p.name}: {p.target(root)}")
        lines += [
            "",
            "A manifest inherited from a project template and never updated parses",
            "cleanly, ports cleanly, and produces a capability that fails only when",
            "somebody calls it. By then the port is 'done' and the failure looks like",
            "the platform's. Fix langgraph.json first, then re-run this.",
            "",
            "This is not a rare case. It is what the first real project tried against",
            "this tool looked like.",
        ]
    lines += [
        "",
        "Each one gains a CLI verb, an MCP tool, an agent-facing function and a",
        "generated SKILL.md from that single registration (ADR-072), plus an audit",
        "record and the caller's identity. None of that needs code.",
        "",
        "How the platform finds them:",
        "",
        f"  axi ext init {distribution}",
        f"  python -m axiom_ext_langgraph.port {manifest.name} \\",
        f"      --namespace {namespace} --into {distribution}/{distribution}",
        "",
        "  --into writes the module into <ext>/skills/ AND wires __init__.py.",
        "  That second step is not optional and is the one that fails silently:",
        "  axi ext init scaffolds skills/ with a .gitkeep and no __init__.py, so",
        "  the directory imports as a namespace package, the platform finds no",
        "  bind_default on it, and returns no capabilities without an error.",
        "",
        "Discovery is by directory, not by entry point. A distribution that is not",
        "laid out as an extension is never scanned, and nothing will say so — the",
        "capabilities simply will not exist.",
        "",
    ]

    decisions: list[str] = []
    in_code = [p.name for p in ports if p.compiles_with_a_checkpointer(root)]
    if in_code:
        decisions.append(
            "  checkpointer in code: "
            + ", ".join(in_code)
            + "\n    compile(checkpointer=...) appears in the graph's own source, which\n"
            "    is where this usually lives rather than in the manifest. That is a\n"
            "    second state store. Ported, the pause belongs on the platform's\n"
            "    ApprovalGate, which is durable and is the single record of what was\n"
            "    allowed; two of them fork that record and a safety case cannot cite\n"
            "    both. Read as text, not imported, so treat it as a prompt to look."
        )
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


def install_into_extension(
    package_dir: Path, module_text: str, namespace: str
) -> list[str]:
    """Write the module into an extension's ``skills`` package and wire it up.

    Returns what it did, for the caller to print.

    The wiring is the part that has to happen and is easy to miss. ``axi ext
    init`` scaffolds ``skills/`` with a ``.gitkeep`` and no ``__init__.py``, so
    the directory imports as a namespace package: the platform's loader does
    ``import_module("<ext>.skills")``, finds no ``bind_default`` on it, and
    returns nothing. No error, no capability, no clue. Dropping a module in
    beside the ``.gitkeep`` is not enough, and telling somebody to "import it
    from ``__init__.py``" is not enough either when the file does not exist.
    """
    done: list[str] = []
    skills = package_dir / "skills"
    skills.mkdir(parents=True, exist_ok=True)

    module_name = f"{namespace}_graphs"
    (skills / f"{module_name}.py").write_text(module_text, encoding="utf-8")
    done.append(f"wrote {skills / (module_name + '.py')}")

    init = skills / "__init__.py"
    existing = init.read_text(encoding="utf-8") if init.is_file() else ""

    if "def bind_default" in existing:
        # This extension already binds its own skills. Appending an import
        # would give the module two functions of that name and the second would
        # win, silently dropping whatever the extension registered before.
        done.append(
            f"NOT wired: {init} already defines bind_default. Call "
            f"{module_name}.register_all(registry) from inside it instead."
        )
        return done

    line = f"from .{module_name} import bind_default, register_all  # noqa: F401"
    if line in existing:
        done.append(f"{init} already imports it")
        return done

    prefix = existing.rstrip("\n") + "\n\n" if existing.strip() else ""
    init.write_text(f"{prefix}{_INIT_MARKER}\n{line}\n", encoding="utf-8")
    done.append(f"{'patched' if existing.strip() else 'created'} {init}")
    return done


def has_missing_targets(manifest: Path) -> bool:
    """Whether any declared graph points at a file that is not there."""
    root = Path(manifest).parent
    return any(p.missing_from(root) for p in read_langgraph_manifest(Path(manifest)))


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
        "--force",
        action="store_true",
        help="generate even when a declared graph file is missing",
    )
    parser.add_argument(
        "--write",
        type=Path,
        default=None,
        help="write the registration module to this path instead of printing it",
    )
    parser.add_argument(
        "--into",
        type=Path,
        default=None,
        help=(
            "an extension's package directory (the one containing skills/). "
            "Writes the module AND wires skills/__init__.py, which is the step "
            "that otherwise fails silently."
        ),
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

    if has_missing_targets(args.manifest) and not args.force:
        print(
            "\nRefusing to generate a registration for a manifest whose graphs are "
            "not there. Pass --force if you are porting ahead of the code.",
            file=sys.stderr,
        )
        return 2

    module = build_registration_module(args.manifest, args.namespace)

    if args.into:
        print()
        for note in install_into_extension(args.into, module, args.namespace):
            print(f"  {note}")
        return 0

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
