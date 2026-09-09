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

    def test_it_prints_both_required_entry_points(self, manifest):
        """One alone silently does nothing."""
        lines = "\n".join(report(manifest(), "moose", "moose"))

        assert "axiom.portfolio_member" in lines
        assert "axiom.skills" in lines

    def test_the_distribution_name_can_differ_from_the_namespace(self, manifest):
        lines = "\n".join(report(manifest(), "moose", "moose-agent"))
        assert 'moose-agent = "moose:__name__"' in lines


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
        out = tmp_path / "pkg" / "axiom_capabilities.py"
        code = main([str(manifest()), "--namespace", "moose", "--write", str(out)])

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
