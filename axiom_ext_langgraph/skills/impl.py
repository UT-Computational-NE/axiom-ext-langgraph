# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""``langgraph.report`` — read a langgraph.json and say what it would become.

A read. It does not register capabilities and it does not compile a graph.
Registration is the ``port`` verb, which writes. Splitting them keeps the
skill on ``risk = "read"``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from axiom_ext_langgraph.graphs import LangGraphManifestError, read_langgraph_manifest

__all__ = ["report", "report_document"]


def report_document(document: str, *, namespace: str) -> dict[str, Any]:
    """Parse ``langgraph.json`` text into the capability report.

    Raises:
        LangGraphManifestError: the document is not a manifest we can read.
        ValueError: ``namespace`` is empty.
    """
    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError("namespace must be a non-empty string")
    if not isinstance(document, str) or not document.strip():
        raise LangGraphManifestError("document must be the text of a langgraph.json")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "langgraph.json"
        path.write_text(document, encoding="utf-8")
        ports = read_langgraph_manifest(path)

    return {
        "namespace": namespace,
        "capabilities": [f"{namespace}.{port.name}" for port in ports],
        "graphs": [
            {"name": port.name, "path": port.path, "entry": port.entry, "description": port.description}
            for port in ports
        ],
    }


def report(params: dict[str, Any], ctx: Any) -> Any:
    """Skill function ``(params, ctx) -> SkillResult``.

    ``ctx`` is accepted because that is the skill signature. The report does
    not read it: a read of a document the caller pasted has no principal to
    consult and no registry to touch.
    """
    del ctx
    from axiom.infra.skills import SkillResult

    try:
        value = report_document(
            str(params.get("document", "")),
            namespace=str(params.get("namespace", "")),
        )
    except (LangGraphManifestError, ValueError) as exc:
        return SkillResult(ok=False, errors=[str(exc)])
    return SkillResult(ok=True, value=value)
