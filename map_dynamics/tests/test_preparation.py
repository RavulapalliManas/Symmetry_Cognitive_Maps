"""Preparation contracts. These tests do not establish scientific validity."""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def api():
    assert importlib.util.find_spec("map_dynamics.preparation") is not None, (
        "The preparation API must be implemented."
    )
    return importlib.import_module("map_dynamics.preparation")


def test_registered_pilot_expands_training_arms_without_expanding_delays():
    plan = api().prepare(ROOT, "X1")
    arms = plan["arms"]
    pilot = yaml.safe_load((ROOT / "registries/experiments.yaml").read_text())["experiments"]["X1"]["pilot"]
    assert len(arms) == pilot["primary_network_count"]
    assert len({arm["run_id"] for arm in arms}) == len(arms)
    for seed in (0, 1, 2):
        paired = [arm for arm in arms if arm["seed"] == seed]
        assert len(paired) == pilot["primary_training_condition_count"]
        assert len({arm["pair_id"] for arm in paired}) == 1
    assert all(arm["status"] == "planned_not_run" for arm in arms)
    assert plan["evaluation_delay_steps"] == pilot["evaluation_delay_steps"]
    assert plan["launchable"] is False
    assert plan["status"] == "draft_not_frozen"
    assert plan["requirements"]
    assert plan["provenance"]["files"]["configs/protocols/X1.yaml"]


def test_cycle_is_rejected():
    registry = {"X0": {"depends_on": ["X1"]}, "X1": {"depends_on": ["X0"]}}
    with pytest.raises(ValueError, match="cycle"):
        api().dependency_order(registry)


def test_missing_dependency_is_rejected():
    with pytest.raises(ValueError, match="unknown dependency"):
        api().dependency_order({"X1": {"depends_on": ["missing"]}})


def test_registry_and_uploaded_source_files_are_consistent():
    result = api().validate(ROOT)
    assert result["dependency_order"] == ["X0", "X1", "X2", "X3", "X4", "X5"]
    assert result["scientific_execution_verified"] is False
    assert set(result["claims"].values()) == {"untested"}


def test_duplicate_seeds_and_unregistered_mode_are_rejected(tmp_path):
    config = yaml.safe_load((ROOT / "configs/protocols/X1.yaml").read_text())
    config["paired_seeds"] = [0, 0, 2]
    with pytest.raises(ValueError, match="unique"):
        api().pilot_arms(config, {"tasks": ["shared_rule", "context_specific_rule"], "learning_modes": ["predictive_only", "joint", "task_only"], "paired_seeds": 3})
    config["paired_seeds"] = [0, 1, 2]
    config["learning_modes"].append("made_up")
    with pytest.raises(ValueError, match="learning_modes"):
        api().pilot_arms(config, {"tasks": ["shared_rule", "context_specific_rule"], "learning_modes": ["predictive_only", "joint", "task_only"], "paired_seeds": 3})


def test_downstream_experiments_keep_unspecified_conditions_unexpanded():
    plan = api().prepare(ROOT, "X5")
    assert plan["arms"] == []
    assert plan["depends_on"] == ["X1", "X2", "X3"]
    assert plan["launchable"] is False


def test_preparation_writes_once_and_never_changes_registries(tmp_path):
    before = (ROOT / "registries/experiments.yaml").read_bytes()
    target = tmp_path / "plan.json"
    command = [sys.executable, "-m", "map_dynamics.cli", "--root", str(ROOT), "prepare", "--experiment", "X1", "--output", str(target)]
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert json.loads(target.read_text())["launchable"] is False
    original = target.read_bytes()
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode != 0
    assert target.read_bytes() == original
    assert (ROOT / "registries/experiments.yaml").read_bytes() == before


def test_source_inventory_distinguishes_present_and_missing_assets(tmp_path):
    (tmp_path / "utils").mkdir()
    (tmp_path / "utils/Architectures.py").write_text("# fixture\n")
    inventory = api().audit_legacy(tmp_path)
    assert inventory[0]["availability"] == "present_not_compatibility_tested"
    assert len(inventory[0]["sha256"]) == 64
    assert any(row["availability"] == "missing" for row in inventory)
    assert all(row["model_loaded"] is False for row in inventory)


def test_cli_has_no_scientific_launch_command():
    result = subprocess.run([sys.executable, "-m", "map_dynamics.cli", "--root", str(ROOT), "run"], capture_output=True, text=True)
    assert result.returncode == 2
    assert "invalid choice" in result.stderr
