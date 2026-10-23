# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""``axi langgraph`` — operator verbs for the toolkit.

``port`` is the existing porter. ``inspect``, ``validate``, ``mermaid``, and
``dry-run`` are :mod:`axiom_ext_langgraph.graphwork`. ``gate`` is
:func:`axiom_ext_langgraph.host_tool.decide_host_call` against the platform
file queue, unless the caller passes a gate. None of these is a second
implementation of the library.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from axiom_ext_langgraph.graphwork import (
    GraphDocumentError,
    dry_run,
    inspect_graph,
    load_graph_document,
    to_mermaid,
    validate_graph,
)
from axiom_ext_langgraph.host_tool import decide_host_call, open_platform_gate
from axiom_ext_langgraph.port import main as port_main

__all__ = ["build_parser", "cli"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="axi langgraph",
        description="Port, inspect, and gate LangGraph work on the Axiom platform.",
    )
    sub = parser.add_subparsers(dest="verb", required=True)

    port = sub.add_parser("port", help="Read langgraph.json and register each graph as a capability.")
    port.add_argument("manifest", type=Path, help="path to langgraph.json")
    port.add_argument("--namespace", required=True, help="capability prefix, conventionally the extension's CLI noun")
    port.add_argument("--distribution", default="", help="distribution name for the entry points (default: the namespace)")
    port.add_argument("--force", action="store_true", help="generate even when a declared graph file is missing")
    port.add_argument("--write", type=Path, default=None, help="write the registration module to this path instead of printing it")
    port.add_argument("--into", type=Path, default=None, help="an extension package directory; writes the module and wires skills/__init__.py")

    for name, help_text in (
        ("inspect", "Report nodes, edges, entries, exits, unreachable nodes, and cycles."),
        ("validate", "inspect, plus whether the graph is structurally sound."),
        ("mermaid", "Write a Mermaid flowchart of the graph."),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("document", type=Path, help="JSON graph document")

    dry = sub.add_parser("dry-run", help="Trace steps along the edges. Does not run node functions.")
    dry.add_argument("document", type=Path, help="JSON graph document")
    dry.add_argument(
        "--choice",
        action="append",
        default=None,
        metavar="NODE=LABEL",
        help="branch to follow at a conditional node; repeat for each node",
    )

    gate = sub.add_parser(
        "gate",
        help="Submit a host call to the platform approval gate. Does not run the call.",
    )
    gate.add_argument("--name", required=True, help="action name the gate classifies")
    gate.add_argument("--params", default="{}", help="JSON object of parameters")
    gate.add_argument("--approval-id", default=None, help="look up an action instead of submitting a new one")
    return parser


def _emit(payload: Any) -> None:
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")


def _choices(pairs: list[str] | None) -> dict[str, str]:
    choices: dict[str, str] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"choice must be node=label, got {pair!r}")
        node, label = pair.split("=", 1)
        if not node or not label:
            raise ValueError(f"choice must be node=label, got {pair!r}")
        if node in choices:
            raise ValueError(f"choice for {node} was given twice")
        choices[node] = label
    return choices


def _port(args: argparse.Namespace) -> int:
    forwarded = [str(args.manifest), "--namespace", args.namespace]
    if args.distribution:
        forwarded += ["--distribution", args.distribution]
    if args.force:
        forwarded.append("--force")
    if args.write is not None:
        forwarded += ["--write", str(args.write)]
    if args.into is not None:
        forwarded += ["--into", str(args.into)]
    return port_main(forwarded)


def cli(argv: list[str] | None = None, *, gate: Any = None) -> int:
    """Run ``axi langgraph``. Returns a process status code.

    ``gate`` is the approval gate for the ``gate`` verb. The default is the
    platform file queue. Tests pass the gate they already hold.
    """
    args = build_parser().parse_args(argv)
    try:
        if args.verb == "port":
            return _port(args)
        if args.verb == "inspect":
            _emit(inspect_graph(load_graph_document(args.document)))
            return 0
        if args.verb == "validate":
            report = validate_graph(load_graph_document(args.document))
            _emit(report)
            return 0 if report["ok"] else 1
        if args.verb == "mermaid":
            sys.stdout.write(to_mermaid(load_graph_document(args.document)))
            return 0
        if args.verb == "dry-run":
            _emit(dry_run(load_graph_document(args.document), choices=_choices(args.choice)))
            return 0
        if args.verb == "gate":
            params = json.loads(args.params)
            if not isinstance(params, dict):
                raise TypeError("params must be a JSON object")
            used = gate if gate is not None else open_platform_gate()
            decision = decide_host_call(used, args.name, params, approval_id=args.approval_id)
            _emit(decision.as_dict())
            return 0 if decision.allowed else 3
    except (GraphDocumentError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled verb {args.verb!r}")
