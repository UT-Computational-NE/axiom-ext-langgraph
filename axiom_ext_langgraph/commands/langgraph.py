# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""``axi langgraph`` — operator verbs for the toolkit.

The only verb is ``port``. It is the existing porter
(:func:`axiom_ext_langgraph.port.main`) with a noun in front, not a second
implementation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from axiom_ext_langgraph.port import main as port_main

__all__ = ["build_parser", "cli"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="axi langgraph",
        description="Port a LangGraph project onto the one Axiom skill registry.",
    )
    sub = parser.add_subparsers(dest="verb", required=True)
    port = sub.add_parser(
        "port",
        help="Read langgraph.json and register each graph as a capability.",
    )
    port.add_argument("manifest", type=Path, help="path to langgraph.json")
    port.add_argument(
        "--namespace",
        required=True,
        help="capability prefix, conventionally the extension's CLI noun",
    )
    port.add_argument(
        "--distribution",
        default="",
        help="distribution name for the entry points (default: the namespace)",
    )
    port.add_argument(
        "--force",
        action="store_true",
        help="generate even when a declared graph file is missing",
    )
    port.add_argument(
        "--write",
        type=Path,
        default=None,
        help="write the registration module to this path instead of printing it",
    )
    port.add_argument(
        "--into",
        type=Path,
        default=None,
        help="an extension package directory; writes the module and wires skills/__init__.py",
    )
    return parser


def cli(argv: list[str] | None = None) -> int:
    """Run ``axi langgraph``. Returns a process status code."""
    args = build_parser().parse_args(argv)
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
