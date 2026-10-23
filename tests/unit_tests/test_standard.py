# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""Standard AEOS conformance tests.

Inherits the base classes shipped in ``axiom-tests`` (AEOS §8.2). That
package is the conformance plugin, not the platform. CI installs it
from the axiom-os commit named in ``.github/workflows/ci.yml``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

axiom_unit = pytest.importorskip(
    "axiom_tests.unit_tests",
    reason="axiom-tests is not installed; pip install -e '.[test]'",
)
ExtensionStandardTests = axiom_unit.ExtensionStandardTests
CommandTests = axiom_unit.CommandTests
SkillTests = axiom_unit.SkillTests
ToolTests = axiom_unit.ToolTests

_ROOT = Path(__file__).resolve().parents[2]


def _manifest() -> dict:
    with (_ROOT / "axiom-extension.toml").open("rb") as handle:
        return tomllib.load(handle)


def _provides(kind: str) -> dict:
    for block in _manifest()["extension"]["provides"]:
        if block.get("kind") == kind:
            return block
    raise AssertionError(f"no provides block of kind {kind}")


class TestAxiomExtLanggraphStandard(ExtensionStandardTests):
    @pytest.fixture
    def extension_manifest_path(self) -> Path:
        return _ROOT / "axiom-extension.toml"

    @property
    def require_docs_dir(self) -> bool:
        return True

    @property
    def require_agents_md(self) -> bool:
        return True


class TestProjectSkillsTool(ToolTests):
    @pytest.fixture
    def tool_class(self) -> type:
        from axiom_ext_langgraph.tools.project_skills import ProjectSkills

        return ProjectSkills

    @pytest.fixture
    def tool_manifest_block(self) -> dict:
        return _provides("tool")

    @property
    def idempotent(self) -> bool:
        return True

    @property
    def side_effects(self) -> str:
        return "none"


class TestLanggraphCommand(CommandTests):
    @pytest.fixture
    def command_entry(self):
        from axiom_ext_langgraph.commands.langgraph import cli

        return cli

    @pytest.fixture
    def command_manifest_block(self) -> dict:
        return _provides("cmd")

    @property
    def expected_noun(self) -> str:
        return "langgraph"

    @property
    def expected_subcommands(self) -> tuple[str, ...]:
        return ("port",)


class TestReportSkill(SkillTests):
    @pytest.fixture
    def skill_path(self) -> Path:
        return _ROOT / "axiom_ext_langgraph" / "skills" / "report"
