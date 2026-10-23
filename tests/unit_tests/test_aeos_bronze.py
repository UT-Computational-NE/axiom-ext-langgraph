# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""Bronze checks from ``axi ext lint`` that do not need the platform CLI.

The directory-name rule (AEOS030) compares ``Path.name`` to the manifest
name with no hyphen folding. A GitHub checkout is named ``axiom-ext-langgraph``.
The check below lint the same tree through a path whose last component is
the manifest name, which is what the linter does.
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_REQUIRED = (
    "README.md",
    "CHANGELOG.md",
    "LICENSE",
    "pyproject.toml",
    "axiom-extension.toml",
)
_KIND_DIRS = {
    "agent": "agents",
    "tool": "tools",
    "cmd": "commands",
    "service": "services",
    "adapter": "adapters",
    "skill": "skills",
    "hook": "hooks",
}
_MCP_OPT_OUT = re.compile(
    "^\\s*#\\s*mcp\\s*:\\s*not[- ]applicable\\s*[-\\u2014\\u2013]\\s*\\S+",
    re.MULTILINE,
)
_POSTURES = {"carries", "magnitude-only", "none", "not-applicable"}
_ENTRY_GROUPS = {"tool": "axiom.tools", "cmd": "axiom.commands"}


def _load(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _declares_all(source: str) -> bool:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    return True
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "__all__":
            return True
    return False


def test_required_files_and_docs_and_agents_symlink():
    missing = [name for name in _REQUIRED if not (_ROOT / name).is_file()]
    assert not missing
    assert (_ROOT / "docs" / "prds" / "prd.md").is_file()
    assert (_ROOT / "docs" / "specs" / "spec.md").is_file()
    assert (_ROOT / "tests" / "unit_tests" / "test_standard.py").is_file()
    claude = _ROOT / "CLAUDE.md"
    assert claude.is_symlink()
    assert claude.resolve() == (_ROOT / "AGENTS.md").resolve()


def test_manifest_matches_pyproject_and_entry_points():
    manifest = _load(_ROOT / "axiom-extension.toml")
    pyproject = _load(_ROOT / "pyproject.toml")
    ext = manifest["extension"]
    project = pyproject["project"]
    assert ext["aeos_version"] == "0.1.0"
    assert ext["version"] == project["version"]
    assert ext["description"] == project["description"]
    assert project["name"].replace("-", "_") == ext["name"]
    assert project["requires-python"] == ext["compatibility"]["python"]
    assert ext["provides"]

    posture = ext["uncertainty"]["posture"]
    assert posture in _POSTURES

    text = (_ROOT / "axiom-extension.toml").read_text(encoding="utf-8")
    assert _MCP_OPT_OUT.search(text)

    groups = pyproject["project"]["entry-points"]
    for block in ext["provides"]:
        group = _ENTRY_GROUPS.get(block["kind"])
        if group is None:
            continue
        key = block.get("name") or block.get("noun")
        assert groups[group][key] == block["entry"]


def test_package_declares_public_api_and_kind_directories():
    manifest = _load(_ROOT / "axiom-extension.toml")
    name = manifest["extension"]["name"]
    init = _ROOT / name / "__init__.py"
    assert init.is_file()
    assert _declares_all(init.read_text(encoding="utf-8"))
    provided = {block["kind"] for block in manifest["extension"]["provides"]}
    for kind, dirname in _KIND_DIRS.items():
        if kind in provided:
            assert (_ROOT / name / dirname).is_dir(), dirname
    for block in manifest["extension"]["provides"]:
        if block.get("kind") == "skill":
            assert (_ROOT / block["path"] / "SKILL.md").is_file()


def test_lint_directory_name_matches_when_the_path_is_the_manifest_name(tmp_path):
    manifest_name = _load(_ROOT / "axiom-extension.toml")["extension"]["name"]
    link = tmp_path / manifest_name
    link.symlink_to(_ROOT)
    assert link.name == manifest_name
    assert (link / manifest_name / "__init__.py").is_file()
    assert (link / "axiom-extension.toml").is_file()


def test_manifest_validates_against_the_aeos_schema():
    axiom_manifest = pytest.importorskip(
        "axiom_tests._manifest",
        reason="axiom-tests is not installed; pip install -e '.[test]'",
    )
    manifest = _load(_ROOT / "axiom-extension.toml")
    errors = axiom_manifest.validate_manifest(manifest)
    assert not errors, "\n".join(errors)
