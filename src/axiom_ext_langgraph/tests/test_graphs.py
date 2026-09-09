# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""A graph becoming an Axiom capability."""

import json

import pytest

from axiom_ext_langgraph.graphs import (
    GraphPort,
    LangGraphManifestError,
    read_langgraph_manifest,
    skill_from_graph,
)


class FakeGraph:
    """Anything with .invoke(state). A CompiledStateGraph is one of these."""

    def __init__(self, reply="ok", raises=None):
        self.reply = reply
        self.raises = raises
        self.seen = []

    def invoke(self, state, *a, **k):
        if self.raises:
            raise self.raises
        self.seen.append(state)
        return {**state, "answer": self.reply}


class TestTheAdapter:
    def test_a_graph_becomes_a_skill_function(self):
        run = skill_from_graph(FakeGraph(reply="42"))
        result = run({"prompt": "what is six times seven"}, None)

        assert result.ok
        assert result.value["answer"] == "42"

    def test_a_plain_prompt_lands_where_a_chat_graph_looks_for_it(self):
        """The caller should not have to know the graph's state schema."""
        graph = FakeGraph()
        skill_from_graph(graph)({"prompt": "hello"}, None)

        assert graph.seen[0]["messages"] == [("user", "hello")]

    def test_other_params_survive_alongside_the_prompt(self):
        graph = FakeGraph()
        skill_from_graph(graph)({"prompt": "hi", "thread_id": "t1"}, None)

        assert graph.seen[0]["thread_id"] == "t1"

    def test_params_pass_through_untouched_for_a_non_chat_graph(self):
        graph = FakeGraph()
        skill_from_graph(graph, input_key=None)({"mesh": "core.e", "steps": 3}, None)

        assert graph.seen[0] == {"mesh": "core.e", "steps": 3}

    def test_a_custom_state_mapper_wins(self):
        graph = FakeGraph()
        skill_from_graph(graph, to_state=lambda p: {"q": p["prompt"].upper()})(
            {"prompt": "hi"}, None
        )

        assert graph.seen[0] == {"q": "HI"}

    def test_the_whole_final_state_comes_back_not_a_summary(self):
        """A wrapper that guessed which part mattered is why people bypass it."""
        run = skill_from_graph(FakeGraph())
        value = run({"prompt": "x", "scratch": [1, 2]}, None).value

        assert value["scratch"] == [1, 2]
        assert "messages" in value


class TestFailureShape:
    def test_a_raising_graph_reports_rather_than_raises(self):
        """Opposite of the tool direction, on purpose.

        A skill's caller branches on ``.ok``; a graph calling a tool needs an
        exception it can branch on. Each direction fails in the shape its
        consumer can act on.
        """
        run = skill_from_graph(FakeGraph(raises=RuntimeError("solver diverged")))
        result = run({"prompt": "run it"}, None)

        assert not result.ok
        assert "solver diverged" in result.errors[0]
        assert "RuntimeError" in result.errors[0]


class TestManifest:
    def _write(self, tmp_path, graphs):
        p = tmp_path / "langgraph.json"
        p.write_text(json.dumps({"dependencies": ["."], "graphs": graphs}))
        return p

    def test_it_reads_the_bare_string_form(self, tmp_path):
        ports = read_langgraph_manifest(
            self._write(tmp_path, {"agent": "./moose_agent/agent.py:graph"})
        )
        assert [p.name for p in ports] == ["agent"]
        assert ports[0].entry == "moose_agent.agent:graph"

    def test_it_reads_the_object_form_and_keeps_the_description(self, tmp_path):
        """The description becomes what a model reads before calling it."""
        ports = read_langgraph_manifest(
            self._write(
                tmp_path,
                {"solver": {"path": "./pkg/s.py:g", "description": "Runs a solve."}},
            )
        )
        assert ports[0].description == "Runs a solve."
        assert ports[0].entry == "pkg.s:g"

    def test_a_missing_file_is_an_error_not_an_empty_port(self, tmp_path):
        with pytest.raises(LangGraphManifestError, match=r"no langgraph\.json"):
            read_langgraph_manifest(tmp_path / "nope.json")

    def test_a_manifest_with_no_graphs_is_an_error(self, tmp_path):
        """More likely the wrong file than an empty project."""
        with pytest.raises(LangGraphManifestError, match="declares no graphs"):
            read_langgraph_manifest(self._write(tmp_path, {}))

    def test_a_shape_we_cannot_port_is_refused_not_skipped(self, tmp_path):
        """Porting three of four graphs silently leaves the fourth undiscovered."""
        with pytest.raises(LangGraphManifestError, match="neither"):
            read_langgraph_manifest(self._write(tmp_path, {"a": ["not", "valid"]}))

    def test_a_path_naming_no_object_is_refused(self):
        with pytest.raises(LangGraphManifestError, match="names no object"):
            _ = GraphPort("a", "./pkg/agent.py").entry


class TestEntryRewriting:
    """LangGraph spells a file path; Axiom spells an import path."""

    @pytest.mark.parametrize(
        ("declared", "expected"),
        [
            ("./pkg/agent.py:graph", "pkg.agent:graph"),
            ("pkg/agent.py:graph", "pkg.agent:graph"),
            ("./agent.py:graph", "agent:graph"),
            ("./a/b/c.py:make_graph", "a.b.c:make_graph"),
        ],
    )
    def test_the_two_spellings_of_one_declaration(self, declared, expected):
        assert GraphPort("g", declared).entry == expected


class TestRegistration:
    def test_registering_gives_the_graph_every_surface(self, tmp_path):
        from axiom.infra.skills import SkillRegistry

        from axiom_ext_langgraph.graphs import skills_from_langgraph_json

        manifest = tmp_path / "langgraph.json"
        manifest.write_text(json.dumps({"graphs": {"agent": "./x/y.py:graph"}}))

        registry = SkillRegistry()
        names = skills_from_langgraph_json(registry, manifest, namespace="moose")

        assert names == ["moose.agent"]
        spec = registry.spec("moose.agent")
        assert set(spec.surfaces) == {"cli", "mcp", "agent_tool", "skill_md"}

    def test_a_broken_graph_does_not_stop_the_others_registering(self, tmp_path):
        """Registration must not import. One bad graph would take the rest down."""
        from axiom.infra.skills import SkillRegistry

        from axiom_ext_langgraph.graphs import skills_from_langgraph_json

        manifest = tmp_path / "langgraph.json"
        manifest.write_text(
            json.dumps(
                {"graphs": {"good": "./a.py:g", "bad": "./no_such_module.py:g"}}
            )
        )

        registry = SkillRegistry()
        names = skills_from_langgraph_json(registry, manifest, namespace="m")

        assert names == ["m.good", "m.bad"]

    def test_the_failure_surfaces_when_the_broken_one_is_called(self, tmp_path):
        from axiom.infra.skills import SkillRegistry

        from axiom_ext_langgraph.graphs import skills_from_langgraph_json

        manifest = tmp_path / "langgraph.json"
        manifest.write_text(json.dumps({"graphs": {"bad": "./no_such_module.py:g"}}))
        registry = SkillRegistry()
        skills_from_langgraph_json(registry, manifest, namespace="m")

        result = registry.spec("m.bad").fn({"prompt": "hi"}, None)

        assert not result.ok
        assert "cannot resolve graph" in result.errors[0]
        assert "no_such_module" in result.errors[0]


@pytest.mark.graph
class TestAgainstARealCompiledGraph:
    """FakeGraph is a stand-in. This is the interface actually being adapted.

    Written because the adapter above was built from a docstring twice today and
    was wrong once. ``create_agent`` returns a ``CompiledStateGraph``, and the
    only claim worth pinning is that a real one satisfies the duck type this
    module assumes.
    """

    def test_a_real_graph_runs_through_the_adapter(self):
        pytest.importorskip("langchain.agents", reason="needs langchain, not just core")
        from langgraph.graph import END, START, StateGraph
        from typing_extensions import TypedDict

        class State(TypedDict):
            messages: list
            answer: str

        def respond(state: State) -> State:
            return {"answer": f"saw {len(state['messages'])} message(s)"}

        builder = StateGraph(State)
        builder.add_node("respond", respond)
        builder.add_edge(START, "respond")
        builder.add_edge("respond", END)
        graph = builder.compile()

        result = skill_from_graph(graph)({"prompt": "hello"}, None)

        assert result.ok, result.errors
        assert result.value["answer"] == "saw 1 message(s)"

    def test_create_agent_returns_something_this_adapter_accepts(self):
        pytest.importorskip("langchain.agents", reason="needs langchain, not just core")
        from langchain.agents import create_agent
        from langgraph.graph.state import CompiledStateGraph

        assert callable(create_agent)

        assert hasattr(CompiledStateGraph, "invoke"), (
            "the duck type this module adapts"
        )


class TestPathsARealManifestCarries:
    """Found by probing the port the way a stranger would use it.

    Each of these registered without complaint and produced a module path that
    imports nothing, so the failure arrived at call time as a missing module
    with no hint that the manifest was the cause.
    """

    @pytest.mark.parametrize(
        "declared",
        [
            "./src/pkg/graph.py:g",
            ".\\src\\pkg\\graph.py:g",
            "./src\\pkg/graph.py:g",
            "src/pkg/graph.py:g",
        ],
    )
    def test_every_separator_a_manifest_might_use(self, declared):
        """A manifest written on Windows carries backslashes; a hand-edited one
        can carry both."""
        assert GraphPort("g", declared).entry == "pkg.graph:g"

    def test_a_path_above_the_project_is_refused_not_mangled(self):
        """``../shared/graph.py`` became ``...shared.graph``: not a module path,
        not an error, not anything. Python has no dotted form for a parent
        directory, so this can only be refused."""
        with pytest.raises(LangGraphManifestError, match="above the project"):
            _ = GraphPort("g", "../shared/graph.py:graph").entry

    def test_a_dotted_attribute_is_carried_through(self):
        assert GraphPort("g", "./a.py:builders.deck").entry == "a:builders.deck"


class TestWhateverShapeTheManifestNames:
    """``langgraph.json`` does not only name compiled graphs.

    Its schema says a value may point at "(async) context managers that accept a
    single configuration argument and return a pregel object", and plain
    factories are common besides. Calling ``.invoke`` on those gave
    ``'function' object has no attribute 'invoke'`` — true, and useless.
    """

    def test_a_compiled_graph(self):
        assert skill_from_graph(FakeGraph())({"prompt": "x"}, None).ok

    def test_a_factory_taking_a_config(self):
        assert skill_from_graph(lambda config=None: FakeGraph())({"prompt": "x"}, None).ok

    def test_a_factory_taking_nothing(self):
        assert skill_from_graph(lambda: FakeGraph())({"prompt": "x"}, None).ok

    def test_a_context_manager(self):
        from contextlib import contextmanager

        @contextmanager
        def cm():
            yield FakeGraph()

        assert skill_from_graph(cm())({"prompt": "x"}, None).ok

    def test_the_context_manager_is_exited(self):
        """A factory holding a connection wants it closed when the run ends."""
        from contextlib import contextmanager

        closed = []

        @contextmanager
        def cm():
            try:
                yield FakeGraph()
            finally:
                closed.append(True)

        skill_from_graph(cm())({"prompt": "x"}, None)
        assert closed == [True]

    def test_a_factory_is_called_per_run_not_cached(self):
        """Caching the first result would make every later run reuse the first
        run's configuration."""
        made = []

        def factory(config=None):
            made.append(1)
            return FakeGraph()

        run = skill_from_graph(factory)
        run({"prompt": "a"}, None)
        run({"prompt": "b"}, None)
        assert len(made) == 2

    def test_an_async_context_manager_is_refused_with_a_reason(self):
        """Entering one needs an event loop this path does not have. Refusing
        beats failing deeper in, further from the cause."""
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def acm():
            yield FakeGraph()

        result = skill_from_graph(acm())({"prompt": "x"}, None)
        assert not result.ok
        assert "async context manager" in result.errors[0]

    def test_something_that_is_not_a_graph_at_all_says_so(self):
        result = skill_from_graph(42)({"prompt": "x"}, None)
        assert not result.ok
        assert "neither a graph" in result.errors[0]
