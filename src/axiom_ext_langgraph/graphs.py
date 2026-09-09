# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""The other direction: a graph becomes an Axiom capability.

``tools.py`` projects a skill *into* a graph as a leaf tool. Until now nothing
went the other way, and the asymmetry cost more than it looked like it did: a
graph was a graph and nothing else. It had no CLI verb, no MCP tool, no
SKILL.md, no audit record, and no identity beyond whatever the process happened
to have bound.

This module closes it. Wrap a compiled graph, register it with a ``SkillSpec``,
and ADR-072 projection hands you every one of those surfaces from the single
declaration. The graph's author writes a graph and gets platform-native
behaviour without writing platform code.

The shapes already line up
--------------------------

``langgraph.json`` declares ``graphs`` as a map of name to ``"path/to/file.py:
object"``. Axiom's ``SkillRegistry.register_entry`` takes a name and a
``"module.path:function"`` entry, resolved lazily on first call. That is the
same declaration written twice, which is why :func:`skills_from_langgraph_json`
can read one and produce the other rather than asking anyone to hand-port a
manifest.

What a graph is, in Axiom's terms
---------------------------------

A skill is ``(params, ctx) -> SkillResult``. A compiled graph is
``invoke(state, config) -> state``. The adapter is small, and the three things
it must not get wrong are:

**Identity.** The run is bound with ``acting_as`` from the skill context's
principal, so every gateway call inside the graph is attributed to whoever
invoked the capability rather than to the host process.

**Failure.** A graph that raises produces ``ok=False`` with the error, because a
skill's caller branches on ``.ok``. This is the opposite of the tool direction,
where a failure *raises* so a graph sees an error it can branch on rather than
a string that reads like an answer. Each direction fails in the shape its
consumer can act on.

**State.** The final state is returned, not summarized. A caller that wants the
last message can take it; a caller that wants the scratchpad can have it. A
wrapper that guessed which part mattered would be the lossy hop that makes
people bypass it.

What this deliberately does not do
----------------------------------

It does not install a checkpointer. A graph that pauses should pause on the
platform's ``ApprovalGate``, which is durable, is the single record of what was
allowed, and survives the process. Handing the graph LangGraph's checkpointer
here would fork that record, and a safety case cannot cite two.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from axiom_ext_langgraph.actor import acting_as

__all__ = [
    "GraphPort",
    "LangGraphManifestError",
    "read_langgraph_manifest",
    "skill_from_graph",
    "skills_from_langgraph_json",
]

log = logging.getLogger(__name__)

#: Where a chat-shaped graph takes its input. ``create_agent`` graphs use this.
DEFAULT_INPUT_KEY = "messages"

#: Directory name that means "source root", not "package". Stripped when
#: rewriting a declared file path into an import path.
SOURCE_ROOT = "src"


class LangGraphManifestError(ValueError):
    """A ``langgraph.json`` could not be read, or says something we cannot port.

    Raised rather than skipped. A migration tool that silently ports three of
    four graphs leaves the fourth undiscovered until somebody notices a missing
    verb, and by then the port is "done".
    """


def _principal_of(ctx: Any) -> Any:
    return getattr(ctx, "principal", None)


def skill_from_graph(
    graph: Any,
    *,
    input_key: str | None = DEFAULT_INPUT_KEY,
    to_state: Callable[[Mapping[str, Any]], Any] | None = None,
) -> Callable[[dict[str, Any], Any], Any]:
    """Adapt a compiled graph into a skill function.

    Args:
        graph: anything with ``.invoke(state)`` — a ``CompiledStateGraph``,
            which is what both ``create_agent`` and ``StateGraph.compile``
            return.
        input_key: for a chat-shaped graph, the state key a plain ``prompt``
            param is placed under as one user message. ``None`` disables that
            convenience and passes params through unchanged.
        to_state: full control over params-to-state, for a graph whose state
            schema is not chat-shaped. Overrides ``input_key``.

    Returns:
        ``(params, ctx) -> SkillResult``, ready for a ``SkillSpec``.
    """
    from axiom.infra.skills import SkillResult

    def run(params: dict[str, Any], ctx: Any) -> Any:
        if to_state is not None:
            state = to_state(params)
        elif input_key is not None and "prompt" in params:
            # The one convenience worth having: a capability whose caller types
            # a sentence should not have to know the graph's state schema.
            state = {input_key: [("user", params["prompt"])], **{
                k: v for k, v in params.items() if k != "prompt"
            }}
        else:
            state = dict(params)

        try:
            with acting_as(_principal_of(ctx)):
                final = graph.invoke(state)
        except Exception as exc:
            log.exception("graph capability failed")
            return SkillResult(ok=False, errors=[f"{type(exc).__name__}: {exc}"])

        return SkillResult(ok=True, value=final)

    return run


class GraphPort:
    """One graph named in a ``langgraph.json``, and where it points.

    Deliberately not a dataclass of strings: ``entry`` is the thing that has to
    be exactly right for the port to work, and keeping the raw declaration
    beside it means a failed resolution can show what it was asked to resolve.
    """

    def __init__(self, name: str, path: str, description: str = "") -> None:
        self.name = name
        self.path = path
        self.description = description

    @property
    def entry(self) -> str:
        """The declaration rewritten as an Axiom lazy entry, ``module:attr``.

        ``./pkg/agent.py:graph`` becomes ``pkg.agent:graph``. The file-path form
        is LangGraph's; Axiom resolves an import path, and the two differ only
        in spelling.

        A leading ``src/`` is dropped, because ``src`` is a source root rather
        than a package in the layout that uses it: ``./src/react_agent/graph.py``
        is imported as ``react_agent.graph``. This is not hypothetical tidiness.
        The first real project tried against this tool is a src-layout one, and
        without the strip every ported capability would resolve to a module that
        does not exist.

        The strip happens even when ``src/__init__.py`` is present, which it
        sometimes is by accident. Packaging decides what is importable, and no
        packaging configuration in the wild exposes ``src`` itself as a package.
        """
        file_part, _, attr = self.path.partition(":")
        if not attr:
            raise LangGraphManifestError(
                f"graph {self.name!r} declares {self.path!r}, which names no object. "
                "LangGraph's form is 'path/to/file.py:object'."
            )
        parts = [
            part
            for part in Path(file_part).with_suffix("").as_posix().split("/")
            if part not in ("", ".")
        ]
        if parts and parts[0] == SOURCE_ROOT:
            parts = parts[1:]
        if not parts:
            raise LangGraphManifestError(
                f"graph {self.name!r} declares {self.path!r}, which names no module."
            )
        return f"{'.'.join(parts)}:{attr}"

    def target(self, root: Path) -> Path:
        """Where the declared file should be, relative to the project root."""
        file_part, _, _ = self.path.partition(":")
        return (root / file_part).resolve()

    def compiles_with_a_checkpointer(self, root: Path) -> bool:
        """Whether the declared source compiles the graph with a checkpointer.

        A heuristic, and said to be one wherever it is reported. It reads the
        declared file for ``checkpointer=`` rather than importing it, because
        importing a research project's graph module pulls in its whole
        dependency tree and often its credentials.

        It exists because the manifest's own ``checkpointer`` key is not where
        this usually lives. The first real project tried against this tool
        declares nothing in the manifest and does
        ``builder.compile(checkpointer=memory)`` on line 274, which is the
        second state store the port most needs to talk about, and a
        manifest-only check sails straight past it.
        """
        target = self.target(root)
        if not target.exists():
            return False
        try:
            return "checkpointer=" in target.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False

    def missing_from(self, root: Path) -> bool:
        """Whether the declaration points at a file that is not there.

        Worth checking at port time rather than leaving to first call. A
        manifest inherited from a project template and never updated still
        parses, still ports, and produces a capability that fails only when
        somebody invokes it — by which point the port is "done" and the failure
        looks like ours.
        """
        return not self.target(root).exists()

    def __repr__(self) -> str:
        return f"GraphPort({self.name!r}, {self.path!r})"


def read_langgraph_manifest(path: str | Path) -> list[GraphPort]:
    """Read a ``langgraph.json`` and return the graphs it declares.

    Handles both value forms the schema allows: a bare ``"file.py:obj"`` string
    and a ``{"path": ..., "description": ...}`` object. The description is worth
    carrying because it becomes the ``SkillSpec`` description, which is what a
    model sees when deciding whether to call the capability, and writing it
    twice is how the two drift.
    """
    import json

    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LangGraphManifestError(f"no langgraph.json at {p}") from exc
    except json.JSONDecodeError as exc:
        raise LangGraphManifestError(f"{p} is not valid JSON: {exc}") from exc

    graphs = raw.get("graphs")
    if not isinstance(graphs, dict) or not graphs:
        raise LangGraphManifestError(
            f"{p} declares no graphs. Nothing to port, which is more likely a "
            "wrong file than an empty project."
        )

    ports: list[GraphPort] = []
    for name, value in graphs.items():
        if isinstance(value, str):
            ports.append(GraphPort(name, value))
        elif isinstance(value, dict) and "path" in value:
            ports.append(GraphPort(name, value["path"], value.get("description", "")))
        else:
            raise LangGraphManifestError(
                f"graph {name!r} in {p} is neither a 'file.py:obj' string nor an "
                f"object with a 'path' key; got {type(value).__name__}"
            )
    return ports


def skills_from_langgraph_json(
    registry: Any,
    manifest: str | Path,
    *,
    namespace: str,
    surfaces: tuple[str, ...] = ("cli", "mcp", "agent_tool", "skill_md"),
) -> list[str]:
    """Register every graph a ``langgraph.json`` declares as an Axiom skill.

    Args:
        registry: an ``axiom.infra.skills.SkillRegistry``.
        manifest: path to the ``langgraph.json``.
        namespace: the qualified-name prefix, conventionally the extension's
            CLI noun. A graph named ``agent`` in namespace ``moose`` becomes
            the capability ``moose.agent``.
        surfaces: which projections these capabilities opt into. The default is
            everything, which is right for a port — the point is that the graph
            gains surfaces — but a graph that writes should have its surfaces
            narrowed deliberately, the same way a decision verb is CLI-only.

    Returns:
        The capability names registered.
    """
    from axiom.infra.skills import SkillSpec

    registered: list[str] = []
    for port in read_langgraph_manifest(manifest):
        capability = f"{namespace}.{port.name}"
        registry.register_skill(
            SkillSpec(
                name=capability,
                fn=_lazy_graph_skill(port),
                description=port.description or f"LangGraph capability {port.name}.",
                inputs={"prompt": "str"},
                surfaces=surfaces,
            )
        )
        registered.append(capability)
    return registered


def _lazy_graph_skill(port: GraphPort) -> Callable[[dict[str, Any], Any], Any]:
    """Defer importing the graph until the capability is actually called.

    A port registers every graph in the manifest, and importing them all at
    registration would make one broken graph take down the whole extension's
    registration — including the graphs that are fine.
    """
    import importlib

    resolved: dict[str, Any] = {}

    def run(params: dict[str, Any], ctx: Any) -> Any:
        from axiom.infra.skills import SkillResult

        if "fn" not in resolved:
            module_path, _, attr = port.entry.partition(":")
            try:
                module = importlib.import_module(module_path)
                graph = getattr(module, attr)
            except Exception as exc:
                return SkillResult(
                    ok=False,
                    errors=[
                        f"cannot resolve graph {port.name!r} from {port.path!r} "
                        f"(as {port.entry!r}): {type(exc).__name__}: {exc}"
                    ],
                )
            resolved["fn"] = skill_from_graph(graph)
        return resolved["fn"](params, ctx)

    return run
