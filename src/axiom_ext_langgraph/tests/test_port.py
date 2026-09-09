# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""The one-command port, and the decisions it refuses to make."""

import json

import pytest

from axiom_ext_langgraph.port import build_registration_module, main, report


@pytest.fixture
def manifest(tmp_path):
    def write(**extra):
        p = tmp_path / "langgraph.json"
        p.write_text(
            json.dumps(
                {
                    "dependencies": ["."],
                    "graphs": {"agent": "./moose_agent/agent.py:graph"},
                    **extra,
                }
            )
        )
        return p

    return write


class TestTheReport:
    def test_it_shows_both_spellings_of_each_graph(self, manifest):
        lines = "\n".join(report(manifest(), "moose", "moose"))

        assert "./moose_agent/agent.py:graph" in lines
        assert "moose.agent" in lines
        assert "moose_agent.agent:graph" in lines

    def test_it_explains_how_the_platform_finds_them(self, manifest):
        """This used to print two entry points, one of which does not exist.

        ``axiom.skills`` is read by nothing. Following the old instructions
        produced a port that succeeded and capabilities that never registered.
        """
        lines = "\n".join(report(manifest(), "moose", "moose"))

        assert "axi ext init" in lines
        assert "bind_default" in lines
        assert "by directory, not by entry point" in lines

    def test_the_distribution_name_is_what_scaffolds_the_extension(self, manifest):
        lines = "\n".join(report(manifest(), "moose", "moose-agent"))
        assert "axi ext init moose-agent" in lines


class TestDecisionsItRefusesToMake:
    """A tool that silently defaults these has made them badly."""

    def test_a_checkpointer_is_flagged_not_translated(self, manifest):
        lines = "\n".join(
            report(manifest(checkpointer={"ttl": {"strategy": "delete"}}), "m", "m")
        )

        assert "checkpointer" in lines
        assert "ApprovalGate" in lines
        assert "fork that record" in lines

    def test_no_checkpointer_means_no_checkpointer_advice(self, manifest):
        """Advice that always fires is advice nobody reads."""
        lines = "\n".join(report(manifest(), "m", "m"))
        assert "ApprovalGate" not in lines

    def test_auth_is_flagged_when_present(self, manifest):
        lines = "\n".join(report(manifest(auth={"path": "./auth.py:auth"}), "m", "m"))
        assert "auth:" in lines

    def test_surfaces_are_always_flagged(self, manifest):
        """The default is right for a read and wrong for a write, every time."""
        lines = "\n".join(report(manifest(), "m", "m"))
        assert "WRITES" in lines


class TestTheGeneratedModule:
    def test_it_reads_the_manifest_rather_than_restating_it(self, manifest):
        """The manifest stays the single declaration; this is regenerated."""
        module = build_registration_module(manifest(), "moose")

        assert "skills_from_langgraph_json" in module
        assert "agent.py:graph" not in module, "graphs must not be copied in"

    def test_it_records_the_command_that_made_it(self, manifest):
        module = build_registration_module(manifest(), "moose")
        assert "python -m axiom_ext_langgraph.port" in module

    def test_the_generated_module_is_valid_python(self, manifest):
        compile(build_registration_module(manifest(), "moose"), "generated", "exec")

    def test_it_finds_the_manifest_from_inside_the_package(self, tmp_path):
        """The layout every real project has: manifest at root, module in the package.

        The first template did ``__file__.parent / "langgraph.json"``, which is
        the sibling form and is wrong exactly there.
        """
        import importlib.util
        import sys

        root = tmp_path / "proj"
        pkg = root / "moose_agent"
        pkg.mkdir(parents=True)
        (root / "langgraph.json").write_text(
            json.dumps({"graphs": {"agent": "./moose_agent/agent.py:graph"}})
        )
        target = pkg / "axiom_capabilities.py"
        target.write_text(build_registration_module(root / "langgraph.json", "moose"))

        spec = importlib.util.spec_from_file_location("gen_caps", target)
        module = importlib.util.module_from_spec(spec)
        sys.modules["gen_caps"] = spec.name and module
        spec.loader.exec_module(module)

        from axiom.infra.skills import SkillRegistry

        registry = SkillRegistry()
        assert module.register_all(registry) == ["moose.agent"]
        assert registry.spec("moose.agent") is not None

    def test_it_says_so_when_the_manifest_is_gone(self, tmp_path):
        import importlib.util

        pkg = tmp_path / "orphan"
        pkg.mkdir()
        target = pkg / "axiom_capabilities.py"
        target.write_text(build_registration_module(tmp_path / "langgraph.json", "m"))

        spec = importlib.util.spec_from_file_location("orphan_caps", target)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with pytest.raises(FileNotFoundError, match="Regenerate"):
            module.register_all(object())


class TestCli:
    def test_it_writes_the_module_when_asked(self, manifest, tmp_path, capsys):
        m = manifest()
        target = m.parent / "moose_agent" / "agent.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("graph = None\n")

        out = tmp_path / "pkg" / "axiom_capabilities.py"
        code = main([str(m), "--namespace", "moose", "--write", str(out)])

        assert code == 0
        assert out.exists()
        assert "register_all" in out.read_text()

    def test_a_bad_manifest_exits_nonzero_with_a_reason(self, tmp_path, capsys):
        bad = tmp_path / "langgraph.json"
        bad.write_text("{}")

        assert main([str(bad), "--namespace", "m"]) == 1
        assert "declares no graphs" in capsys.readouterr().err

    def test_a_missing_manifest_does_not_traceback(self, tmp_path, capsys):
        assert main([str(tmp_path / "nope.json"), "--namespace", "m"]) == 1
        assert "cannot port" in capsys.readouterr().err


class TestWhatTheFirstRealProjectLookedLike:
    """Both checks below exist because a real repo had both problems.

    The public proxy for the project this tool was built for declares
    ``./src/react_agent/graph.py:graph`` in a manifest inherited from the
    LangGraph react-agent template, while its actual code lives in
    ``src/mooseagent/``, and it compiles its graph with a checkpointer in code
    rather than in the manifest. Neither was caught by the first version.
    """

    def _project(self, tmp_path, declared, *, create=None, source=""):
        root = tmp_path / "proj"
        (root / "src" / "pkg").mkdir(parents=True)
        (root / "langgraph.json").write_text(json.dumps({"graphs": {"g": declared}}))
        if create:
            target = root / create
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source)
        return root / "langgraph.json"

    def test_a_src_layout_path_becomes_an_importable_module(self, tmp_path):
        """``src`` is a source root, not a package."""
        from axiom_ext_langgraph.graphs import read_langgraph_manifest

        m = self._project(tmp_path, "./src/pkg/graph.py:graph", create="src/pkg/graph.py")
        assert read_langgraph_manifest(m)[0].entry == "pkg.graph:graph"

    def test_a_manifest_pointing_at_nothing_stops_the_port(self, tmp_path, capsys):
        """It parses, it ports, and it fails only when somebody calls it."""
        m = self._project(tmp_path, "./src/gone/graph.py:graph")

        assert main([str(m), "--namespace", "x"]) == 2
        assert "Refusing to generate" in capsys.readouterr().err

    def test_the_report_names_the_file_that_is_not_there(self, tmp_path):
        m = self._project(tmp_path, "./src/gone/graph.py:graph")
        lines = "\n".join(report(m, "x", "x"))

        assert "MISSING" in lines
        assert "STOP." in lines
        assert "src/gone/graph.py" in lines

    def test_force_ports_ahead_of_the_code(self, tmp_path):
        m = self._project(tmp_path, "./src/gone/graph.py:graph")
        assert main([str(m), "--namespace", "x", "--force"]) == 0

    def test_a_checkpointer_compiled_in_code_is_flagged(self, tmp_path):
        """The manifest key is not where this usually lives."""
        m = self._project(
            tmp_path,
            "./src/pkg/graph.py:graph",
            create="src/pkg/graph.py",
            source="graph = builder.compile(checkpointer=memory)\n",
        )
        lines = "\n".join(report(m, "x", "x"))

        assert "checkpointer in code" in lines
        assert "ApprovalGate" in lines

    def test_a_graph_without_one_is_not_flagged(self, tmp_path):
        """Advice that always fires is advice nobody reads."""
        m = self._project(
            tmp_path,
            "./src/pkg/graph.py:graph",
            create="src/pkg/graph.py",
            source="graph = builder.compile()\n",
        )
        assert "checkpointer in code" not in "\n".join(report(m, "x", "x"))

    def test_the_detection_reads_rather_than_imports(self, tmp_path):
        """Importing a research graph pulls its whole dependency tree."""
        m = self._project(
            tmp_path,
            "./src/pkg/graph.py:graph",
            create="src/pkg/graph.py",
            source="import nonexistent_dependency\ngraph = b.compile(checkpointer=m)\n",
        )
        assert "checkpointer in code" in "\n".join(report(m, "x", "x"))


class TestTheHookIsTheOneThePlatformActuallyCalls:
    """The port used to print an entry point that does not exist.

    It told people to add ``[project.entry-points."axiom.skills"]``. Nothing
    reads that group. Following the instructions exactly would have produced a
    port that succeeded, an entry point that did nothing, and capabilities that
    never registered — with nothing anywhere saying so.

    The real mechanism is AEOS layout: the platform imports an extension's
    ``skills`` package and calls ``bind_default()`` if it is there.
    """

    def test_the_generated_module_exposes_bind_default(self, manifest):
        """The name is the contract. Rename it and nothing registers, silently."""
        module = build_registration_module(manifest(), "moose")
        assert "def bind_default(" in module

    def test_bind_default_binds_into_the_process_default_registry(self, tmp_path):
        """``skills_emit`` calls it with no arguments and reads the default
        registry afterwards. Anything else registers into a registry nobody
        looks at."""
        import importlib.util

        from axiom.infra.skills import default_registry

        root = tmp_path / "proj"
        (root / "pkg").mkdir(parents=True)
        (root / "langgraph.json").write_text(
            json.dumps({"graphs": {"solver": "./pkg/graph.py:g"}})
        )
        (root / "pkg" / "graph.py").write_text("g = None\n")
        target = root / "pkg" / "caps.py"
        target.write_text(build_registration_module(root / "langgraph.json", "portns"))

        spec = importlib.util.spec_from_file_location("portns_caps", target)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        returned = module.bind_default()

        assert returned is default_registry()
        assert default_registry().spec("portns.solver") is not None

    def test_it_does_not_advertise_an_entry_point_that_nothing_reads(self, manifest):
        lines = "\n".join(report(manifest(), "moose", "moose"))
        assert "axiom.skills" not in lines

    def test_it_says_discovery_is_by_directory(self, manifest):
        """The failure mode is silent, so the instructions have to name it."""
        lines = "\n".join(report(manifest(), "moose", "moose"))
        assert "axi ext init" in lines
        assert "bind_default" in lines


class TestWiringTheExtension:
    """``--into`` exists because the manual version fails silently.

    ``axi ext init`` scaffolds ``skills/`` with a ``.gitkeep`` and no
    ``__init__.py``. Dropping a module in beside it leaves the directory
    importing as a namespace package with no ``bind_default`` on it, and the
    platform's loader returns no capabilities without raising. Walking the
    documented steps by hand produced exactly that, which is how this was found.
    """

    def _ext(self, tmp_path):
        pkg = tmp_path / "demoagent" / "demoagent"
        (pkg / "skills").mkdir(parents=True)
        (pkg / "skills" / ".gitkeep").write_text("")
        return pkg

    def test_it_writes_the_module_and_creates_the_init(self, tmp_path, manifest):
        from axiom_ext_langgraph.port import install_into_extension

        pkg = self._ext(tmp_path)
        install_into_extension(
            pkg, build_registration_module(manifest(), "moose"), "moose"
        )

        assert (pkg / "skills" / "moose_graphs.py").is_file()
        assert "from .moose_graphs import bind_default" in (
            pkg / "skills" / "__init__.py"
        ).read_text()

    def test_bind_default_is_reachable_on_the_package(self, tmp_path):
        """The property the platform's loader actually checks."""
        import importlib.util
        import sys

        from axiom_ext_langgraph.port import install_into_extension

        root = tmp_path / "proj"
        pkg = root / "demoagent"
        (pkg / "skills").mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        (root / "langgraph.json").write_text(
            json.dumps({"graphs": {"deck": "./demoagent/g.py:graph"}})
        )
        (pkg / "g.py").write_text("graph = None\n")
        install_into_extension(
            pkg,
            build_registration_module(root / "langgraph.json", "deckns"),
            "deckns",
        )

        sys.path.insert(0, str(root))
        try:
            spec = importlib.util.spec_from_file_location(
                "wired_skills", pkg / "skills" / "__init__.py"
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules["wired_skills"] = module
            spec.loader.exec_module(module)
            assert hasattr(module, "bind_default")
        finally:
            sys.path.remove(str(root))
            sys.modules.pop("wired_skills", None)

    def test_it_appends_rather_than_clobbering_an_existing_init(self, tmp_path, manifest):
        from axiom_ext_langgraph.port import install_into_extension

        pkg = self._ext(tmp_path)
        (pkg / "skills" / "__init__.py").write_text("EXISTING = 1\n")
        install_into_extension(
            pkg, build_registration_module(manifest(), "moose"), "moose"
        )

        text = (pkg / "skills" / "__init__.py").read_text()
        assert "EXISTING = 1" in text
        assert "from .moose_graphs import" in text

    def test_it_refuses_to_shadow_an_extension_that_binds_its_own(self, tmp_path, manifest):
        """Two bind_defaults in one module means the second wins and whatever
        the extension registered before is silently dropped."""
        from axiom_ext_langgraph.port import install_into_extension

        pkg = self._ext(tmp_path)
        (pkg / "skills" / "__init__.py").write_text("def bind_default():\n    ...\n")

        notes = install_into_extension(
            pkg, build_registration_module(manifest(), "moose"), "moose"
        )

        assert any("NOT wired" in n for n in notes)
        assert "from .moose_graphs import" not in (
            pkg / "skills" / "__init__.py"
        ).read_text()

    def test_running_it_twice_is_idempotent(self, tmp_path, manifest):
        from axiom_ext_langgraph.port import install_into_extension

        pkg = self._ext(tmp_path)
        module = build_registration_module(manifest(), "moose")
        install_into_extension(pkg, module, "moose")
        install_into_extension(pkg, module, "moose")

        text = (pkg / "skills" / "__init__.py").read_text()
        assert text.count("from .moose_graphs import") == 1
