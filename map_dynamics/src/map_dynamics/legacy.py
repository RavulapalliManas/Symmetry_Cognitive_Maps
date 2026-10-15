"""A checked adapter around pinned legacy code. The cell is never reimplemented."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch

from map_dynamics.preparation import file_hash


class LegacyModules:
    def __init__(self, root: Path):
        self.root = root.resolve()
        if not (self.root / "utils/Architectures.py").is_file():
            raise ValueError(f"Not a legacy checkout: {self.root}")
        sys.path.insert(0, str(self.root))
        self.architectures = self._import("utils.Architectures")
        self.cells = self._import("utils.thetaRNN")
        self.headings = self._import("project5_symmetry.environments.hd_encodings")
        self.loader = self._import("project5_symmetry.analysis.run_spectrum")
        self.arenas = self._import("project5_symmetry.environments.arena")

    def _import(self, name):
        module = importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(self.root):
            raise ValueError(f"Legacy import came from another source tree: {name}")
        return module

    def fingerprint(self):
        return {
            str(path.relative_to(self.root)): file_hash(path)
            for package in ("utils", "project5_symmetry")
            for path in sorted((self.root / package).rglob("*.py"))
        }

    def assert_fingerprint(self, expected):
        if self.fingerprint() != expected:
            raise ValueError("Legacy source fingerprint changed")

    def load_checkpoint(self, checkpoint):
        required = {"obs_size", "k", "hidden_size", "trunc", "dropout_p", "hidden_init_sigma", "pred_offset", "condition", "hd_mode", "arena_size", "noise_std"}
        missing = required - checkpoint.get("meta", {}).keys()
        if missing:
            raise ValueError(f"Checkpoint metadata is incomplete: {sorted(missing)}")
        meta = checkpoint["meta"]
        if meta["hd_mode"] not in self.headings.MODES or meta["condition"] != "s2":
            raise ValueError("This reference slice requires an audited s2 heading condition")
        if meta["obs_size"] != 147 or meta["arena_size"] != 18:
            raise ValueError("The reference slice requires the declared legacy arena and view")
        return self.loader.model_from_checkpoint(checkpoint, torch.device("cpu"))

    def heading_contract(self):
        return {
            "axis_half_turn_invariant": self.headings.is_invariant("axis", 2),
            "parity_half_turn_invariant": self.headings.is_invariant("parity", 2),
            "matrices": {name: self.headings._matrix(name).tolist() for name in self.headings.MODES},
            "flip_matrix_verified": False,
            "flip_note": "No alias is installed until the source-manuscript version is reconciled.",
        }


class LegacyAdapter:
    """Complete recurrent state is one hidden tensor for this selected cell."""
    def __init__(self, model):
        self.model = model.eval()
        if model.rnn.cell.__class__.__name__ != "LayerNormRNNCellEager":
            raise ValueError("The complete-state contract has not been checked for this cell")

    @torch.no_grad()
    def step(self, observation, motion, state, noise):
        value = torch.cat([observation, motion], dim=-1)
        activity, complete = self.model.rnn.cell(value, noise, (state, 0))
        if complete[1] != 0:
            raise ValueError("An unrecorded recurrent state component is present")
        return activity, complete[0]

    @torch.no_grad()
    def record(self, observations, motion, initial_state, noise):
        if observations.shape[:2] != motion.shape[:2] or noise.shape[:2] != motion.shape[:2]:
            raise ValueError("Observation, motion, and noise timing differ")
        state = initial_state.clone()
        states = []
        for time in range(motion.shape[1]):
            activity, state = self.step(observations[:, time], motion[:, time], state, noise[:, time])
            states.append(activity.clone())
        return torch.stack(states, dim=1)

    @torch.no_grad()
    def predict(self, observations, motion, initial_state, noise):
        return self.model(observations, motion, state=initial_state.clone().unsqueeze(0), noise_main=noise)
