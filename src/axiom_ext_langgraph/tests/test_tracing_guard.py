# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""The guard exists because langsmith is a *required* transitive dependency of
langchain-core, and it activates by environment variable alone — one export in
a shell or a service unit and the process ships question text and retrieved
chunks to a hosted third party. No code change, no import, no review.

So the contract under test is: importing this package leaves no LangChain-family
environment configuration alive in the process. Not a blocklist of today's
variables — langsmith grows new ones every release — but the prefix class,
with the master switches pinned explicitly off and one loud, deliberate escape
hatch for a reviewed decision.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from axiom_ext_langgraph._tracing_guard import (
    ALLOW_ENV,
    MASTER_SWITCHES,
    enforce_no_external_tracing,
)


def test_empty_env_gets_master_switches_pinned_false() -> None:
    env: dict[str, str] = {}
    neutralized = enforce_no_external_tracing(env)
    assert neutralized == []
    for switch in MASTER_SWITCHES:
        assert env[switch] == "false"


def test_live_config_is_removed_and_named() -> None:
    env = {
        "LANGSMITH_API_KEY": "ls-secret",
        "LANGSMITH_TRACING": "true",
        "LANGCHAIN_PROJECT": "reactor-manuals",
        "PATH": "/usr/bin",
    }
    neutralized = enforce_no_external_tracing(env)
    assert neutralized == [
        "LANGCHAIN_PROJECT",
        "LANGSMITH_API_KEY",
        "LANGSMITH_TRACING",
    ]
    assert "LANGSMITH_API_KEY" not in env
    assert "LANGCHAIN_PROJECT" not in env
    assert env["LANGSMITH_TRACING"] == "false"
    assert env["PATH"] == "/usr/bin"


def test_variables_that_do_not_exist_yet_die_too() -> None:
    # The class fix: a variable langsmith invents next release is covered
    # today, because the rule is the prefix, not an inventory.
    env = {"LANGSMITH_SHINY_NEW_2027_FEATURE": "on"}
    neutralized = enforce_no_external_tracing(env)
    assert neutralized == ["LANGSMITH_SHINY_NEW_2027_FEATURE"]
    assert "LANGSMITH_SHINY_NEW_2027_FEATURE" not in env


def test_escape_hatch_leaves_env_alone() -> None:
    env = {ALLOW_ENV: "1", "LANGSMITH_API_KEY": "ls-deliberate"}
    neutralized = enforce_no_external_tracing(env)
    assert neutralized is None
    assert env["LANGSMITH_API_KEY"] == "ls-deliberate"
    assert "LANGSMITH_TRACING" not in env


def test_escape_hatch_requires_a_truthy_value() -> None:
    env = {ALLOW_ENV: "0", "LANGSMITH_API_KEY": "ls-secret"}
    neutralized = enforce_no_external_tracing(env)
    assert neutralized == ["LANGSMITH_API_KEY"]
    assert "LANGSMITH_API_KEY" not in env


def test_importing_the_package_neutralizes_the_real_process_env() -> None:
    # The whole claim, executed: a process with live LangSmith config imports
    # the package and the config is gone before any langchain code can act.
    probe = (
        "import axiom_ext_langgraph, json, os; "
        "print(json.dumps({k: v for k, v in os.environ.items() "
        "if k.startswith(('LANGSMITH_', 'LANGCHAIN_'))}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        env={
            **os.environ,
            "LANGSMITH_API_KEY": "ls-secret",
            "LANGSMITH_TRACING": "true",
            "LANGCHAIN_TRACING_V2": "true",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    survivors = json.loads(result.stdout)
    assert "LANGSMITH_API_KEY" not in survivors
    assert survivors["LANGSMITH_TRACING"] == "false"
    assert survivors["LANGCHAIN_TRACING_V2"] == "false"
