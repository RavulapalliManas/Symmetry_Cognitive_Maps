"""Bounded CPU development of X1, not an authorized scientific campaign.

Peak planning bound: 512 MiB runtime reserve + 64 * B * T * (H + 157)
* 4 bytes for recurrent/predictive graphs and recordings + 32 * P bytes for
parameters, gradients, and Adam state. B=4*families; P is counted from the model.
Arrays are bounded before allocation. The managed fleet command must also set
MemoryMax=2G. One run directory has one writer; files stream per update/checkpoint.
"""
from __future__ import annotations

import json
import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from map_dynamics.diagnostics import C2World, decode, family_split
from map_dynamics.legacy import LegacyModules
from map_dynamics.preparation import file_hash
from map_dynamics.replay import write_json
from map_dynamics.transient import TransientTask


MODES = {"predictive_only", "joint", "task_only", "predictive_stop_gradient_actor", "frozen_population"}
PREDICTIVE = {"predictive_only", "joint", "predictive_stop_gradient_actor"}


class ActiveModel(nn.Module):
    """Unchanged legacy pRNN cell/rollouts with a versioned input and new heads."""
    def __init__(self, legacy, hidden_size, prediction_horizon):
        super().__init__()
        self.backbone = legacy.architectures.pRNN_th(
            obs_size=147, act_size=10, hidden_size=hidden_size,
            k=prediction_horizon, trunc=200, dropp=0,
            cell=legacy.cells.LayerNormRNNCellEager,
        )
        self.actor = nn.Linear(hidden_size, 2)
        self.critic = nn.Linear(hidden_size, 1)

    def activity(self, observations, signals, initial, noise):
        return self.backbone._forward_main_trajectory(
            observations, signals, state=initial, noise_main=noise)

    def predict(self, activity, observations, signals):
        count = observations.shape[1] - self.backbone.k
        if count < 1:
            raise ValueError("Episode is shorter than the prediction horizon")
        anchors = torch.arange(count)
        prediction = self.backbone._rollout_from_anchors(activity, signals, anchors)
        target = torch.stack([observations[:, offset:offset+count]
                              for offset in range(self.backbone.k+1)], dim=1)
        return prediction, target.detach()


class ForcedRouteLearner:
    """PPO on the final response only; forced movement has no actor loss.

    Every PPO epoch replays the full episode from its saved reset state and
    sampled noise. There is no cached hidden-state approximation or burn-in.
    Trials end after one response, so the value target is its terminal reward.
    """
    def __init__(self, legacy, task, *, mode="joint", hidden_size=32,
                 prediction_horizon=5, seed=9, noise_std=.03, hidden_init_sigma=.1,
                 learning_rate=.001, prediction_weight=1., value_weight=.5,
                 entropy_weight=.01, ppo_clip=.2, ppo_epochs=2, gradient_clip=1.):
        if mode not in MODES:
            raise ValueError("Unknown learning mode")
        self.task, self.mode = task, mode
        self.contract = {"task": task.contract(), "mode": mode, "hidden_size": hidden_size,
                         "prediction_horizon": prediction_horizon, "seed": seed,
                         "noise_std": noise_std, "hidden_init_sigma": hidden_init_sigma,
                         "learning_rate": learning_rate, "prediction_weight": prediction_weight,
                         "value_weight": value_weight, "entropy_weight": entropy_weight,
                         "ppo_clip": ppo_clip, "ppo_epochs": ppo_epochs,
                         "gradient_clip": gradient_clip, "advantage_normalization": False,
                         "recurrent_replay": "full_episode_from_reset", "dropout": 0.,
                         "legacy_fingerprint": legacy.fingerprint(),
                         "legacy_runtime": {"hoist_input_projection": bool(legacy.architectures._HOIST_INPUT_PROJ)},
                         "implementation": {name: file_hash(Path(__file__).with_name(name))
                                            for name in ("learning.py", "transient.py")},
                         "torch_version": str(torch.__version__)}
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.model = ActiveModel(legacy, hidden_size, prediction_horizon)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.updates, self.optimizer_steps, self.environment_transitions = 0, 0, 0
        self.protocol = None

    def states(self, batch, generator=None):
        families = len(batch.analyst.family) // 4
        width = self.contract["hidden_size"]
        initial = torch.rand(families, width, generator=generator) * self.contract["hidden_init_sigma"]
        noise = torch.randn(families, batch.agent.obs.shape[1], width, generator=generator)
        return initial.repeat_interleave(4, 0), noise.repeat_interleave(4, 0) * self.contract["noise_std"]

    @torch.no_grad()
    def collect(self, families, delay):
        data_seed = int(self.rng.integers(2**31))
        batch = self.task.batch(families, delay, data_seed)
        obs, signals = torch.from_numpy(batch.agent.obs), torch.from_numpy(batch.agent.motion)
        initial, noise = self.states(batch)
        activity = self.model.activity(obs, signals, initial, noise)
        distribution = Categorical(logits=self.model.actor(activity[:, -1]))
        actions = distribution.sample()
        # The environment evaluates the chosen response. Targets never enter the model.
        rewards = (actions == torch.from_numpy(batch.analyst.target)).float()
        mask = torch.zeros(obs.shape[:2], dtype=torch.bool)
        mask[:, -1] = True
        return {"obs": obs, "signals": signals, "initial": initial, "noise": noise,
                "actions": actions, "rewards": rewards, "old_log_probs": distribution.log_prob(actions),
                "old_values": self.model.critic(activity[:, -1]).squeeze(-1),
                "decision_mask": mask, "policy_version": self.updates, "data_seed": data_seed,
                "terminated": torch.ones(len(actions), dtype=torch.bool)}

    def losses(self, rollout):
        if rollout["policy_version"] != self.updates:
            raise ValueError("Rollout policy version is stale")
        if not rollout["terminated"].all() or not rollout["decision_mask"][:, -1].all():
            raise ValueError("X1 requires complete trials ending in an agent-selected response")
        activity = self.model.activity(rollout["obs"], rollout["signals"], rollout["initial"], rollout["noise"])
        zero = torch.zeros(())
        prediction = actor = critic = entropy = zero
        if self.mode in PREDICTIVE:
            predicted, target = self.model.predict(activity, rollout["obs"], rollout["signals"])
            prediction = F.mse_loss(predicted, target)
        if self.mode != "predictive_only":
            population = activity[:, -1]
            if self.mode in {"predictive_stop_gradient_actor", "frozen_population"}:
                population = population.detach()
            distribution = Categorical(logits=self.model.actor(population))
            ratio = (distribution.log_prob(rollout["actions"]) - rollout["old_log_probs"]).exp()
            advantage = rollout["rewards"] - rollout["old_values"]
            clipped = ratio.clamp(1-self.contract["ppo_clip"], 1+self.contract["ppo_clip"])
            actor = -torch.minimum(ratio*advantage, clipped*advantage).mean()
            critic = F.mse_loss(self.model.critic(population).squeeze(-1), rollout["rewards"])
            entropy = distribution.entropy().mean()
        total = (self.contract["prediction_weight"]*prediction + actor
                 + self.contract["value_weight"]*critic - self.contract["entropy_weight"]*entropy)
        return {"total": total, "prediction": prediction, "actor": actor, "critic": critic, "entropy": entropy}

    def update(self, families, delay):
        rollout = self.collect(families, delay)
        self.model.train()
        for _ in range(self.contract["ppo_epochs"]):
            self.optimizer.zero_grad(set_to_none=True)
            losses = self.losses(rollout)
            if not all(torch.isfinite(value) for value in losses.values()):
                raise FloatingPointError("Nonfinite training loss")
            losses["total"].backward()
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.contract["gradient_clip"], error_if_nonfinite=True)
            self.optimizer.step()
            self.optimizer_steps += 1
        self.updates += 1
        self.environment_transitions += families * 4 * rollout["obs"].shape[1]
        return {"update": self.updates, "optimizer_steps": self.optimizer_steps,
                "environment_transitions": self.environment_transitions, "data_seed": rollout["data_seed"],
                "delay": delay, "sampled_response_accuracy": None if self.mode == "predictive_only" else float(rollout["rewards"].mean()),
                "gradient_norm": float(norm), **{name: float(value.detach()) for name, value in losses.items()}}

    def save(self, path):
        path = Path(path)
        temporary = path.with_suffix(".partial")
        if path.exists():
            raise FileExistsError(path)
        torch.save({"schema_version": 1, "contract": self.contract, "protocol": self.protocol,
                    "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "torch_rng": torch.get_rng_state(), "data_rng": self.rng.bit_generator.state,
                    "updates": self.updates, "optimizer_steps": self.optimizer_steps,
                    "environment_transitions": self.environment_transitions}, temporary)
        temporary.replace(path)

    def load(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["schema_version"] != 1 or checkpoint["contract"] != self.contract:
            raise ValueError("Checkpoint contract differs from this learner")
        if checkpoint["protocol"] != self.protocol:
            raise ValueError("Checkpoint experience protocol differs from this run")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.updates = checkpoint["updates"]
        self.optimizer_steps = checkpoint["optimizer_steps"]
        self.environment_transitions = checkpoint["environment_transitions"]
        self.rng.bit_generator.state = checkpoint["data_rng"]
        torch.set_rng_state(checkpoint["torch_rng"])


@torch.no_grad()
def evaluate(learner, batch, roles, destination):
    """Fixed evidence/noise across checkpoints. Decoder preprocessing fits only fitting families."""
    generator = torch.Generator().manual_seed(1701)
    initial, noise = learner.states(batch, generator)
    obs, signals = torch.from_numpy(batch.agent.obs), torch.from_numpy(batch.agent.motion)
    activity = learner.model.activity(obs, signals, initial, noise)
    features = activity[:, -1].numpy()
    a = batch.analyst
    testing = roles == "testing"
    beliefs = learner.task.context_beliefs(batch.agent)
    choices = learner.model.actor(activity[:, -1]).argmax(-1).numpy()
    result = {"update": learner.updates, "n_testing": int(testing.sum()),
              "bayes_context_accuracy": float(beliefs[testing, -1].max(-1).mean()),
              "bayes_decision_accuracy": 1. if learner.task.rule == "shared_rule" else float(beliefs[testing, -1].max(-1).mean()),
              "policy_accuracy": None if learner.mode == "predictive_only" else float(np.mean(choices[testing] == a.target[testing])),
              "context_linear": decode(features, a.context, roles),
              "context_nonlinear": decode(features, a.context, roles, kind="knn"),
              "refitted_response": decode(features, a.target, roles),
              "memoryless_context": decode(np.concatenate([batch.agent.obs[:, -1], batch.agent.motion[:, -1]], axis=-1), a.context, roles)}
    np.savez_compressed(destination, activity=activity.numpy(), initial=initial.numpy(), noise=noise.numpy())
    return result


def run_development(root, config, output, *, resume=None, stop_after=None):
    """Small CPU engineering run. Caps cannot authorize or expand a scientific campaign."""
    root, output = Path(root), Path(output)
    delays = config["train_delays"] + config["evaluation_delays"]
    integers = ["updates", "hidden_size", "batch_families", "evaluation_families",
                "prediction_horizon", "checkpoint_every", "threads", "ppo_epochs", "max_wall_seconds"]
    nonnegative = ["noise_std", "hidden_init_sigma", "prediction_weight", "value_weight", "entropy_weight"]
    positive = ["learning_rate", "gradient_clip"]
    if (config["purpose"] != "X1_software_development_not_a_scientific_campaign"
            or any(type(config[k]) is not int or config[k] < 1 for k in integers)
            or not config["train_delays"] or not config["evaluation_delays"]
            or any(type(d) is not int or not 1 <= d <= 64 for d in delays)
            or any(not np.isfinite(config[k]) or config[k] < 0 for k in nonnegative)
            or any(not np.isfinite(config[k]) or config[k] <= 0 for k in positive)
            or not 0 < config["ppo_clip"] < 1
            or config["mode"] not in MODES
            or (config["mode"] in PREDICTIVE and config["prediction_weight"] == 0)
            or (config["mode"] != "predictive_only" and config["value_weight"] == 0)
            or len(set(config["evaluation_delays"])) != len(config["evaluation_delays"])
            or config["updates"] > 32 or not 2 <= config["hidden_size"] <= 128
            or config["batch_families"] > 8 or not 6 <= config["evaluation_families"] <= 24
            or config["threads"] > 2 or config["ppo_epochs"] > 4
            or not 0 < config["max_mem_gib"] <= 2 or config["max_wall_seconds"] > 120
            or type(config["cue_duration"]) is not int or not 0 <= config["cue_duration"] <= 4
            or min(config["train_delays"]) + max(1, config["cue_duration"]) <= config["prediction_horizon"]):
        raise ValueError("Configuration exceeds the bounded X1 development protocol")
    if stop_after is not None and (type(stop_after) is not int or not 0 <= stop_after <= config["updates"]):
        raise ValueError("stop_after exceeds the configured update budget")
    torch.set_num_threads(config["threads"])
    width = config["hidden_size"]
    frames = max(delays) + max(1, config["cue_duration"])
    batch_size = 4*max(config["batch_families"], config["evaluation_families"])
    # Upper bound on active model parameters before any allocation.
    parameters = width*(157+width+152)+150
    estimate = 512*2**20 + 64*batch_size*frames*(width+157)*4 + 32*parameters
    if estimate > config["max_mem_gib"] * 2**30:
        raise MemoryError("Development arrays exceed max_mem_gib")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_json(output / "run.json", {"status": "running_development", "config": config,
                                     "estimated_peak_bytes": estimate, "resume": str(resume) if resume else None})
    previous_handler = signal.getsignal(signal.SIGALRM)

    def deadline(signum, frame):
        raise TimeoutError("X1 development wall-time budget exhausted")

    signal.signal(signal.SIGALRM, deadline)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, config["max_wall_seconds"])
    try:
        legacy_root = Path(config["legacy_root"]).expanduser()
        legacy = LegacyModules(legacy_root if legacy_root.is_absolute() else root / legacy_root)
        world = C2World(legacy)
        task = TransientTask(world, **{key: config[key] for key in (
            "cue_duration", "cue_reliability", "start_state", "heading", "rule")})
        learner_keys = ("mode", "hidden_size", "prediction_horizon", "seed", "noise_std", "hidden_init_sigma",
                        "learning_rate", "prediction_weight", "value_weight", "entropy_weight", "ppo_clip", "ppo_epochs", "gradient_clip")
        learner = ForcedRouteLearner(legacy, task, **{key: config[key] for key in learner_keys})
        learner.protocol = config.copy()
        if resume:
            learner.load(resume)
        limit = config["updates"] if stop_after is None else stop_after
        if learner.updates > limit:
            raise ValueError("Checkpoint exceeds the requested update budget")
        provenance = {str(path.relative_to(root)): file_hash(path) for path in sorted((root / "src/map_dynamics").glob("*.py"))}
        write_json(output / "manifest.json", {"config": config, "contract": learner.contract,
                   "source_hashes": provenance, "lock_sha256": file_hash(root / "uv.lock"),
                   "parameter_count": sum(p.numel() for p in learner.model.parameters()),
                   "torch": torch.__version__, "numpy": np.__version__,
                   "scientific_hypotheses_supported": False, "durable_archive": "not_verified"})
        (output / "bank").mkdir()
        (output / "population").mkdir()
        training_ids = []

        def diagnostics():
            for delay in config["evaluation_delays"]:
                # A separate seed namespace prevents evaluation from advancing training RNG.
                batch = task.batch(config["evaluation_families"], delay, np.random.SeedSequence([config["seed"], delay, 20261016]))
                roles = family_split(batch.analyst.family, seed=1701)
                bank_file = output / "bank" / f"delay_{delay}.npz"
                if not bank_file.exists():
                    np.savez_compressed(bank_file, obs=batch.agent.obs, signals=batch.agent.motion,
                        state_ids=batch.analyst.state_ids, context=batch.analyst.context,
                        query=batch.analyst.query, target=batch.analyst.target,
                        family=batch.analyst.family, roles=roles)
                destination = output / "population" / f"update_{learner.updates:06d}_delay_{delay}.npz"
                row = evaluate(learner, batch, roles, destination)
                row["delay"] = delay
                with (output / "diagnostics.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")

        learner.save(output / f"checkpoint_{learner.updates:06d}.pt")
        diagnostics()
        with (output / "metrics.jsonl").open("x") as stream:
            while learner.updates < limit:
                delay = int(learner.rng.choice(config["train_delays"]))
                row = learner.update(config["batch_families"], delay)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                training_ids.append(learner.updates)
                if learner.updates % config["checkpoint_every"] == 0 or learner.updates == limit:
                    learner.save(output / f"checkpoint_{learner.updates:06d}.pt")
                    diagnostics()
        report = {"status": "complete_development_run", "updates": learner.updates,
                  "optimizer_steps": learner.optimizer_steps,
                  "updates_this_run": len(training_ids), "environment_transitions": learner.environment_transitions,
                  "wall_seconds": time.monotonic()-started,
                  "scientific_hypotheses_supported": False, "campaign_authorized": False}
        write_json(output / "run.json", {**report, "config": config})
        return report
    except Exception as error:
        write_json(output / "run.json", {"status": "failed_development_run", "config": config,
                                         "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)
