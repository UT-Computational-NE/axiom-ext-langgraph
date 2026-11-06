# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""Toolkit behaviour, and the library it calls.

Dependencies arrive as parameters. Nothing here needs a live gateway.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from axiom_ext_langgraph.commands.langgraph import cli
from axiom_ext_langgraph.graphs import LangGraphManifestError
from axiom_ext_langgraph.library import surface_name, tools_from_registry
from axiom_ext_langgraph.skills.impl import report, report_document
from axiom_ext_langgraph.surfaces import SURFACES
from axiom_ext_langgraph.toolkit import AxiomToolkit
from axiom_ext_langgraph.tools.project_skills import ProjectSkills


@dataclass
class FakeResult:
    ok: bool = True
    value: Any = "done"
    errors: list = field(default_factory=list)


@dataclass
class FakeSpec:
    name: str
    description: str = ""
    inputs: dict = field(default_factory=dict)
    surfaces: tuple = ()


class FakeRegistry:
    def __init__(self) -> None:
        self._specs = {
            "data.install": FakeSpec(
                "data.install",
                description="Install a dataset.",
                inputs={"target": "Path"},
                surfaces=("cli", "agent_tool"),
            ),
            "telemetry.series": FakeSpec(
                "telemetry.series",
                description="Fetch a series.",
                surfaces=("cli",),
            ),
        }

    def invoke(self, name, params, ctx):
        return FakeResult()

    def spec(self, name):
        return self._specs.get(name)

    def list(self, namespace=None):
        return sorted(self._specs)


def test_the_toolkit_projects_the_named_skills_and_nothing_else():
    registry = FakeRegistry()
    tools = AxiomToolkit(registry=registry, names=["data.install"]).get_tools()
    direct = tools_from_registry(registry, ["data.install"])
    assert [tool.name for tool in tools] == [tool.name for tool in direct] == ["data_install"]


def test_an_unset_name_list_honours_the_agent_tool_opt_in():
    tools = AxiomToolkit(registry=FakeRegistry()).get_tools()
    assert [tool.name for tool in tools] == ["data_install"]


def test_project_skills_uses_the_same_surface_name_as_the_live_toolkit():
    projected = ProjectSkills.invoke(
        {
            "skills": [
                {
                    "name": "data.install",
                    "description": "Install a dataset.",
                    "inputs": {"target": "Path"},
                }
            ]
        }
    )
    assert projected["tools"][0]["name"] == surface_name("data.install")
    assert projected["tools"][0]["arguments"] == [{"name": "target", "description": "Path"}]


def test_project_skills_refuses_a_skill_with_no_name():
    with pytest.raises(TypeError):
        ProjectSkills.invoke({"skills": [{"description": "no name"}]})


def test_report_document_names_capabilities_and_does_not_need_a_registry():
    document = '{"graphs": {"agent": "./pkg/agent.py:graph"}}'
    report = report_document(document, namespace="moose")
    assert report["capabilities"] == ["moose.agent"]
    assert report["graphs"][0]["entry"] == "pkg.agent:graph"


def test_report_wraps_the_document_in_a_skill_result():
    from axiom.infra.skills import SkillResult

    result = report(
        {"document": '{"graphs": {"agent": "./pkg/agent.py:graph"}}', "namespace": "moose"},
        None,
    )
    assert isinstance(result, SkillResult)
    assert result.ok is True
    assert result.value["capabilities"] == ["moose.agent"]


def test_report_document_refuses_an_empty_namespace():
    with pytest.raises(ValueError):
        report_document('{"graphs": {"agent": "./a.py:g"}}', namespace="  ")


def test_report_document_refuses_a_manifest_with_no_graphs():
    with pytest.raises(LangGraphManifestError):
        report_document("{}", namespace="moose")


def test_port_verb_stops_when_a_declared_file_is_missing(tmp_path):
    manifest = tmp_path / "langgraph.json"
    manifest.write_text('{"graphs": {"agent": "./missing.py:graph"}}', encoding="utf-8")
    assert cli(["port", str(manifest), "--namespace", "moose"]) == 2


def test_checkpoint_store_stays_untouched():
    checkpoint = next(surface for surface in SURFACES if "checkpoint" in surface.name)
    assert checkpoint.mode == "never"
