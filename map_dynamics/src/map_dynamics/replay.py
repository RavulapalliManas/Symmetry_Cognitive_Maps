"""One-checkpoint-at-a-time legacy reference evaluation. No training.

Peak estimate: 512 MiB runtime reserve + 8 * n_states * (hidden_size+160)
* 4 bytes for features and decoder copies. Evaluate the estimate from the
actual configuration and checkpoint. Enforce a 2 GiB process cap
with the fleet runner; max_states and max_mem_gib also bound arrays here.
"""
from __future__ import annotations

import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import sklearn
import torch

from map_dynamics.diagnostics import C2World, decode, decode_spatial, family_split
from map_dynamics.legacy import LegacyAdapter, LegacyModules
from map_dynamics.preparation import file_hash


def write_json(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run(config, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    count = config["families"] * 2 * config["steps"]
    if config["families"] < 3 or config["steps"] < 6 or count > config["max_states"]:
        raise ValueError("Diagnostic bank exceeds its state cap or cannot form separate splits")
    if not 0 < config["max_mem_gib"] <= 2:
        raise ValueError("This reference slice requires an explicit memory limit at or below 2 GiB")
    torch.set_num_threads(config["threads"])
    legacy = LegacyModules(Path(config["legacy_root"]).expanduser())
    legacy.assert_fingerprint(config["legacy_fingerprint"])
    world = C2World(legacy)
    if not np.array_equal(world.views, world.views[world.twin]) or not np.array_equal(world.next[world.twin], world.twin[world.next]):
        raise ValueError("The rendered world fails the exact half-turn control")
    output.mkdir(parents=True)
    (output / "bank").mkdir()
    (output / "population").mkdir()
    started = time.time()
    write_json(output / "run.json", {"status": "running", "config": config})
    try:
        family_ids = np.repeat(np.arange(config["families"]), 2*config["steps"])
        roles = family_split(family_ids, config["seed"])
        write_json(output / "splits.json", {"unit": "trajectory_family_with_all_twins_and_timesteps", "families": {str(i): str(roles[np.flatnonzero(family_ids == i)[0]]) for i in np.unique(family_ids)}})
        models, table = [], []
        for model_index, entry in enumerate(config["checkpoints"]):
            checkpoint_path = Path(entry["path"]).expanduser()
            if file_hash(checkpoint_path) != entry["sha256"]:
                raise ValueError("Checkpoint fingerprint changed")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            meta = checkpoint["meta"]
            estimate = 512*2**20 + 8*count*(meta["hidden_size"]+160)*4
            if estimate > config["max_mem_gib"]*2**30:
                raise MemoryError("Recording and decoder estimate exceeds max_mem_gib")
            if config["steps"] <= meta["k"]:
                raise ValueError("The replay must be longer than the prediction horizon")
            model = legacy.load_checkpoint(checkpoint)
            adapter = LegacyAdapter(model)
            destination = output / "population" / f"checkpoint_{model_index}"
            destination.mkdir()
            width = meta["hidden_size"]
            hidden = np.empty((count, width), dtype=np.float32)
            sensory = np.empty((count, 152), dtype=np.float32)
            contexts = np.empty(count, dtype=np.int64)
            local_positions = np.empty((count, 2), dtype=np.float32)
            ceilings = np.empty(count, dtype=float)
            state_error, prediction_sse, prediction_n = 0., 0., 0
            for family_id in range(config["families"]):
                family = world.paired_trajectory(config["steps"], config["seed"]+family_id, meta["hd_mode"])
                agent, analyst = family.agent, family.analyst
                stem = f"family_{family_id}_{meta['hd_mode']}"
                np.savez_compressed(output / "bank" / f"{stem}.agent.npz", obs=agent.obs, motion=agent.motion)
                np.savez_compressed(output / "bank" / f"{stem}.analyst.npz", state_ids=analyst.state_ids, primitive_actions=analyst.primitive_actions, context=analyst.context, positions=analyst.positions)
                generator = torch.Generator().manual_seed(config["seed"]+family_id)
                initial = torch.rand(1, width, generator=generator) * meta["hidden_init_sigma"]
                initial = initial.expand(2, -1).clone()
                noise = torch.randn(1, config["steps"], width, generator=generator) * meta["noise_std"]
                noise = noise.expand(2, -1, -1).clone()
                obs, motion = torch.from_numpy(agent.obs), torch.from_numpy(agent.motion)
                activity = adapter.record(obs[:, :-1], motion, initial, noise)
                with torch.no_grad():
                    native = model._forward_main_trajectory(obs[:, :-1], motion, state=initial.unsqueeze(0), noise_main=noise)
                error = float((activity-native).abs().max())
                state_error = max(state_error, error)
                if error != 0:
                    raise ValueError("Online adapter states differ from the native legacy sequence")
                prediction, _, target = adapter.predict(obs, motion, initial, noise)
                prediction_sse += float(torch.square(prediction-target).sum())
                prediction_n += prediction.numel()
                np.savez_compressed(destination / f"family_{family_id}.npz", hidden=activity.numpy(), initial_hidden=initial.numpy(), noise_main=noise.numpy())
                segment = slice(family_id*2*config["steps"], (family_id+1)*2*config["steps"])
                hidden[segment] = activity.reshape(-1, width).numpy()
                sensory[segment] = np.concatenate([agent.obs[:, :-1], agent.motion], axis=-1).reshape(-1, 152)
                contexts[segment] = analyst.context[:, :-1].reshape(-1)
                positions = analyst.positions[:, :-1].copy()
                positions[analyst.context[:, :-1] == 1] = 19-positions[analyst.context[:, :-1] == 1]
                local_positions[segment] = positions.reshape(-1, 2)
                ceilings[segment] = np.concatenate([world.observer_accuracy(agent.obs[twin], agent.motion[twin], meta["hd_mode"]) for twin in range(2)])
                del activity, native, prediction, target, obs, motion
            diagnostic = {
                "population_linear": decode(hidden, contexts, roles),
                "population_knn": decode(hidden, contexts, roles, kind="knn"),
                "memoryless_linear": decode(sensory, contexts, roles),
                "spatial_linear": decode_spatial(hidden, local_positions, roles),
            }
            result = {
                "checkpoint": str(checkpoint_path), "checkpoint_sha256": entry["sha256"],
                "meta": meta, "n_states": count, "n_families": config["families"],
                "native_state_max_abs_error": state_error,
                "prediction_mse": prediction_sse/prediction_n,
                "bayes_context_accuracy": float(ceilings[roles == "testing"].mean()),
                "diagnostics": diagnostic, "peak_array_estimate_bytes": estimate,
                "complete_recurrent_state": ["hidden"], "state_before_update": "initial_hidden then previous hidden",
                "exact_continuation_available": all(key in checkpoint for key in ("optimizer", "rng_state")),
                "policy_use": {"status": "unavailable", "reason": "The legacy checkpoint has no trained policy head."},
            }
            models.append(result)
            for name, value in diagnostic.items():
                table.append({"checkpoint_sha256": entry["sha256"], "model_seed": meta.get("seed"), "hd_mode": meta["hd_mode"], "analysis": name, "metric": "r2" if name == "spatial_linear" else "accuracy", "value": value.get("accuracy", value.get("r2")), "n_testing": value["n_testing"], "replicate_unit": "network_not_timestep", "independent_networks": 1})
            write_json(output / f"checkpoint_{model_index}.json", result)
            del hidden, sensory, contexts, local_positions, ceilings, model, adapter, checkpoint
        original = config.get("original_trajectory")
        if original:
            with np.load(Path(original).expanduser(), allow_pickle=False) as trajectory:
                expected = np.eye(4)[trajectory["heading"][:-1]]
                if not np.array_equal(expected, trajectory["act_enc"][:, 1:]):
                    raise ValueError("Stored legacy heading timing differs from the declared contract")
            original = {"path": original, "sha256": file_hash(Path(original).expanduser()), "heading_timing": "heading_at_observation_t"}
        with (output / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        report = {
            "status": "complete_reference_evaluation", "models": models,
            "config": config, "original_trajectory": original,
            "scientific_hypotheses_supported": False,
            "limitations": ["Selected legacy reference checkpoints, not the X1 learning experiment.", "Exact identical-input controls are not a learned developmental effect.", "Decoder scores are accessibility bounds, not a proof of absent information or policy use.", "No cross-network uncertainty or confirmatory inference is estimated.", "The original source-manuscript fingerprint remains unreconciled."],
            "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__},
            "wall_seconds": time.time()-started,
        }
        write_json(output / "report.json", report)
        files = {str(path.relative_to(output)): file_hash(path) for path in sorted(output.rglob("*")) if path.is_file() and path.name != "run.json"}
        execution_source = {path.name: file_hash(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
        write_json(output / "manifest.json", {"schema_version": 1, "files": files, "source_files": config["legacy_fingerprint"], "execution_source": execution_source})
        write_json(output / "run.json", {"status": "complete", "wall_seconds": time.time()-started})
        return report
    except BaseException as error:
        write_json(output / "run.json", {"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise
