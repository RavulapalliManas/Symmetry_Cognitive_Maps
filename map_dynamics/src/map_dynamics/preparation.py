"""Draft preparation only. No model imports, training, or scientific scoring.

Memory: read one selected source file at a time; no population arrays are allocated.
The registered files are small text files. Fleet checks retain a 2 GiB process cap.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dependency_order(experiments: dict[str, Any]) -> list[str]:
    """Return prerequisites first. Unknown references and cycles are errors."""
    ordered: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(experiment_id: str) -> None:
        if experiment_id in visiting:
            raise ValueError(f"Dependency cycle at {experiment_id}")
        if experiment_id in visited:
            return
        if experiment_id not in experiments:
            raise ValueError(f"unknown dependency: {experiment_id}")
        visiting.add(experiment_id)
        for dependency in experiments[experiment_id]["depends_on"]:
            visit(dependency)
        visiting.remove(experiment_id)
        visited.add(experiment_id)
        ordered.append(experiment_id)

    for experiment_id in experiments:
        visit(experiment_id)
    return ordered


@dataclass(frozen=True)
class PilotArm:
    run_id: str
    pair_id: str
    layout_id: str
    seed: int
    task: str
    learning_mode: str
    status: str = "planned_not_run"


def pilot_arms(config: dict[str, Any], registered: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand the registered X1 pilot, never its evaluation-delay axis."""
    seeds = config["paired_seeds"]
    if not isinstance(seeds, list) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("paired_seeds must contain nonnegative integers")
    if len(seeds) != len(set(seeds)):
        raise ValueError("paired_seeds must be unique")
    if len(seeds) != registered["paired_seeds"]:
        raise ValueError("paired_seeds count differs from the registered pilot")
    for field in ("tasks", "learning_modes"):
        if config[field] != registered[field]:
            raise ValueError(f"{field} differ from the registered pilot")
    layout = config["development_layout_id"]
    if not isinstance(layout, str) or not layout:
        raise ValueError("development_layout_id must be a nonempty string")
    return [
        asdict(PilotArm(
            run_id=f"X1_{layout}_{task}_{mode}_seed{seed}",
            pair_id=f"X1_{layout}_seed{seed}",
            layout_id=layout,
            seed=seed,
            task=task,
            learning_mode=mode,
        ))
        for seed in seeds
        for task in config["tasks"]
        for mode in config["learning_modes"]
    ]


def validate(root: Path) -> dict[str, Any]:
    """Check preparation consistency, not scientific correctness or readiness."""
    manifest = json.loads((root / "PACKAGE_MANIFEST.json").read_text())
    for name, record in manifest["inputs"].items():
        if file_hash(root / record["path"]) != record["sha256"]:
            raise ValueError(f"Supplied input changed: {name}")
    original = json.loads((root / "docs/PACKAGE_MANIFEST.json").read_text())
    for name, record in original["files"].items():
        if manifest["inputs"][name]["sha256"] != record["sha256"]:
            raise ValueError(f"Original checksum mismatch: {name}")

    experiments = load_yaml(root / "registries/experiments.yaml")["experiments"]
    order = dependency_order(experiments)
    claims = load_yaml(root / "registries/claims.yaml")["claims"]
    for claim_id, claim in claims.items():
        if any(item not in experiments for item in claim["experiments"]):
            raise ValueError(f"Unknown experiment in claim {claim_id}")
        if claim["status"] != "untested":
            raise ValueError("Preparation tools cannot establish scientific support")

    for experiment_id in order:
        config = load_yaml(root / f"configs/protocols/{experiment_id}.yaml")
        if config["experiment_id"] != experiment_id or config["status"] != "preparation_template":
            raise ValueError(f"Invalid preparation identity or status: {experiment_id}")
        for field, value in config.items():
            if field.endswith("_config"):
                load_yaml(root / value)

    config = load_yaml(root / "configs/protocols/X1.yaml")
    pilot = experiments["X1"]["pilot"]
    arms = pilot_arms(config, pilot)
    if len(arms) != pilot["primary_network_count"]:
        raise ValueError("Pilot arm count differs from the registry")
    for field in ("cue_reliability", "evaluation_delay_steps"):
        if config[field] != pilot[field]:
            raise ValueError(f"Pilot {field} differs from the registry")
    environment = load_yaml(root / config["environment_config"])
    if environment["cue"]["reliability"] != pilot["cue_reliability"]:
        raise ValueError("Environment cue reliability differs from the pilot")
    if environment["delay"]["evaluation_steps"] != pilot["evaluation_delay_steps"]:
        raise ValueError("Environment evaluation delays differ from the pilot")
    return {
        "status": "preparation_consistent",
        "dependency_order": order,
        "pilot_training_arm_count": len(arms),
        "claims": {key: value["status"] for key, value in claims.items()},
        "scientific_execution_verified": False,
        "launchable": False,
    }


def prepare(root: Path, experiment_id: str) -> dict[str, Any]:
    validate(root)
    registry = load_yaml(root / "registries/experiments.yaml")
    if experiment_id not in registry["experiments"]:
        raise ValueError(f"Unknown experiment: {experiment_id}")
    experiment = registry["experiments"][experiment_id]
    path = f"configs/protocols/{experiment_id}.yaml"
    config = load_yaml(root / path)
    referenced = {value: load_yaml(root / value) for field, value in config.items() if field.endswith("_config")}
    preparation = load_yaml(root / "registries/preparation.yaml")
    sources = [
        path, "registries/experiments.yaml", "registries/claims.yaml",
        "registries/preparation.yaml", "PACKAGE_MANIFEST.json", "pyproject.toml",
        *referenced,
        *[str(item.relative_to(root)) for item in sorted((root / "src/map_dynamics").glob("*.py"))],
    ]
    if (root / "uv.lock").is_file():
        sources.append("uv.lock")
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "status": "draft_not_frozen",
        "launchable": False,
        "depends_on": experiment["depends_on"],
        "scientific_status": experiment["scientific_status"],
        "config": config,
        "referenced_configs": referenced,
        "arms": pilot_arms(config, experiment["pilot"]) if experiment_id == "X1" else [],
        "evaluation_delay_steps": config.get("evaluation_delay_steps", []),
        "requirements": preparation["requirements"],
        "proposed_resource_limits": preparation["proposed_resource_limits"],
        "provenance": {"files": {item: file_hash(root / item) for item in sources}},
    }


def audit_legacy(source_root: Path) -> list[dict[str, Any]]:
    """Inventory known source paths. Do not import code or unpickle artifacts."""
    if not source_root.is_dir():
        raise ValueError(f"Legacy source directory does not exist: {source_root}")
    paths = (
        "utils/Architectures.py", "utils/thetaRNN.py",
        "project5_symmetry/training/train.py", "project5_symmetry/training/ensemble.py",
        "project5_symmetry/training/dataset.py", "project5_symmetry/environments/hd_encodings.py",
        "project5_symmetry/environments/compartment_arenas.py", "project7_retention/worlds.py",
        "requirements.txt", "requirements_visual_rnn.txt", "LICENSE",
    )
    return [
        {
            "original_path": str(source_root / item),
            "sha256": file_hash(source_root / item) if (source_root / item).is_file() else None,
            "availability": "present_not_compatibility_tested" if (source_root / item).is_file() else "missing",
            "model_loaded": False,
        }
        for item in paths
    ]
