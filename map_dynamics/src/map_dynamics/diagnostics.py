"""Sparse observers, paired legacy inputs, and held-out diagnostic readouts.

World memory: 1296 states x (147 float32 pixels + 4 int64 destinations),
below 1 MB. A family stores only two trajectories. No dense state-pair array.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class BayesFilter:
    """Row-stochastic sparse transitions; likelihoods use permitted inputs only."""
    def __init__(self, transitions, prior):
        self.belief = np.asarray(prior, dtype=float).copy()
        if self.belief.ndim != 1 or np.any(self.belief < 0) or not np.isfinite(self.belief).all() or self.belief.sum() <= 0:
            raise ValueError("Invalid prior")
        self.belief /= self.belief.sum()
        self.transitions = transitions
        for transition in transitions:
            if not issparse(transition) or transition.shape != (len(self.belief), len(self.belief)):
                raise ValueError("Expected a sparse transition for every action")
            if np.any(transition.data < 0) or not np.isfinite(transition.data).all() or not np.allclose(np.asarray(transition.sum(axis=1)).ravel(), 1):
                raise ValueError("Transition must be row stochastic")

    def predict(self, action):
        if isinstance(action, (int, np.integer)):
            if not 0 <= action < len(self.transitions):
                raise ValueError("Unknown action")
            weights = np.eye(len(self.transitions))[action]
        else:
            weights = np.asarray(action, dtype=float)
        if weights.shape != (len(self.transitions),) or np.any(weights < 0) or not np.isfinite(weights).all() or not np.isclose(weights.sum(), 1):
            raise ValueError("Invalid action probabilities")
        self.belief = sum(weight * (transition.T @ self.belief) for weight, transition in zip(weights, self.transitions))
        return self.belief.copy()

    def observe(self, likelihood):
        likelihood = np.asarray(likelihood, dtype=float)
        if likelihood.shape != self.belief.shape or np.any(likelihood < 0) or not np.isfinite(likelihood).all():
            raise ValueError("Invalid observation likelihood")
        posterior = self.belief * likelihood
        if posterior.sum() <= 0:
            raise ValueError("impossible observation under this model")
        self.belief = posterior / posterior.sum()
        return self.belief.copy()


@dataclass(frozen=True)
class AgentInputs:
    obs: np.ndarray
    motion: np.ndarray


@dataclass(frozen=True)
class AnalystMetadata:
    state_ids: np.ndarray
    primitive_actions: np.ndarray
    context: np.ndarray
    positions: np.ndarray


@dataclass(frozen=True)
class PairedTrajectory:
    agent: AgentInputs
    analyst: AnalystMetadata


class C2World:
    """Exact tables from the legacy MiniGrid renderer and transition function."""
    def __init__(self, legacy):
        self.legacy = legacy
        env = legacy.arenas.PixelObsWrapper(legacy.arenas.SymmetryArena(
            shape="square", size=18, U=4, F=7, seed=0, symmetry_condition="s2",
        ), tile_size=1)
        env.reset(seed=0)
        inner = env.unwrapped
        self.states = np.array([(c, r, h) for c, r in sorted(inner.passable_positions) for h in range(4)])
        index = {tuple(state): i for i, state in enumerate(self.states)}
        self.twin = np.array([index[(19-c, 19-r, (h+2) % 4)] for c, r, h in self.states])
        self.context = (self.states[:, 1] > 9).astype(int)
        self.next = np.empty((len(self.states), 4), dtype=np.int64)
        self.views = np.empty((len(self.states), 147), dtype=np.float32)
        try:
            for i, (c, r, h) in enumerate(self.states):
                inner.agent_pos, inner.agent_dir = np.array([c, r]), int(h)
                self.views[i] = env.observation(inner.gen_obs())["image"].reshape(-1).astype(np.float32) / 255
                for action in range(4):
                    inner.agent_pos, inner.agent_dir = np.array([c, r]), int(h)
                    inner.step(action)
                    self.next[i, action] = index[(*inner.agent_pos, inner.agent_dir)]
        finally:
            env.close()
        self.transitions = [csr_matrix((np.ones(len(self.states)), (np.arange(len(self.states)), self.next[:, action])), shape=(len(self.states), len(self.states))) for action in range(4)]
        self.action_probs = np.array(legacy.arenas.PAPER_ACTION_PROBS[:4], dtype=float)
        self.action_probs /= self.action_probs.sum()

    def paired_trajectory(self, steps, seed, hd_mode):
        rng = np.random.default_rng(seed)
        first = rng.choice(np.flatnonzero(self.context == 0))
        actions = rng.choice(4, size=steps, p=self.action_probs)
        states = np.empty((2, steps+1), dtype=np.int64)
        states[:, 0] = first, self.twin[first]
        for time, action in enumerate(actions):
            states[:, time+1] = self.next[states[:, time], action]
        motion = np.empty((2, steps, 5), dtype=np.float32)
        motion[:, :, 0] = actions == 2
        matrix = self.legacy.headings._matrix(hd_mode)
        motion[:, :, 1:] = matrix.T[self.states[states[:, :-1], 2]]
        return PairedTrajectory(
            AgentInputs(self.views[states].copy(), motion),
            AnalystMetadata(states, np.broadcast_to(actions, (2, steps)).copy(), self.context[states], self.states[states, :2]),
        )

    def observer_accuracy(self, observations, motion, hd_mode):
        """Legacy sees SpeedHD. Hidden turns are marginalized, not supplied."""
        observer = BayesFilter(self.transitions, np.ones(len(self.states)))
        heading_code = self.legacy.headings._matrix(hd_mode).T[self.states[:, 2]]
        result = []
        for time, signal in enumerate(motion):
            if time:
                weights = self.action_probs * ((np.arange(4) == 2) == bool(motion[time-1, 0]))
                observer.predict(weights / weights.sum())
            matches = np.all(self.views == observations[time], axis=1)
            matches &= np.all(heading_code == signal[1:], axis=1)
            belief = observer.observe(matches.astype(float))
            result.append(max(belief[self.context == context].sum() for context in (0, 1)))
        return np.asarray(result)


def family_split(families, seed):
    families = np.asarray(families)
    unique = np.unique(families)
    if len(unique) < 3:
        raise ValueError("Separate fitting, validation, and testing require at least three families")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    roles = np.empty(len(families), dtype="U10")
    for role, group in zip(("fitting", "validation", "testing"), np.array_split(unique, 3)):
        roles[np.isin(families, group)] = role
    return roles


def decode(features, target, roles, evaluation_features=None, kind="linear"):
    """Fixed hyperparameters. All fitted preprocessing uses the fitting split."""
    features, target, roles = np.asarray(features), np.asarray(target), np.asarray(roles)
    if features.ndim != 2 or len(features) != len(target) or len(roles) != len(target) or not np.isfinite(features).all():
        raise ValueError("Invalid decoder arrays")
    fitting, testing = roles == "fitting", roles == "testing"
    if not fitting.any() or not testing.any() or len(np.unique(target[fitting])) < 2:
        raise ValueError("Decoder requires fitting classes and a separate test split")
    if kind not in ("linear", "knn"):
        raise ValueError("Unknown decoder kind")
    classifier = LogisticRegression(C=1, max_iter=1000, random_state=0) if kind == "linear" else KNeighborsClassifier(n_neighbors=5, weights="distance")
    model = make_pipeline(StandardScaler(), classifier)
    model.fit(features[fitting], target[fitting])
    evaluated = features if evaluation_features is None else np.asarray(evaluation_features)
    if evaluated.shape != features.shape or not np.isfinite(evaluated).all():
        raise ValueError("Fixed-decoder evaluation features differ in shape or contain nonfinite values")
    predictions = model.predict(evaluated[testing])
    scaler, classifier = model.steps[0][1], model.steps[1][1]
    return {
        "accuracy": float(np.mean(predictions == target[testing])),
        "n_fitting": int(fitting.sum()), "n_testing": int(testing.sum()),
        "estimator": "StandardScaler+" + type(classifier).__name__, "kind": kind,
        "preprocessing_hash": hashlib.sha256(scaler.mean_.tobytes() + scaler.scale_.tobytes()).hexdigest(),
        "readout": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "coef": classifier.coef_.tolist() if kind == "linear" else None, "intercept": classifier.intercept_.tolist() if kind == "linear" else None, "parameters": classifier.get_params()},
    }


def decode_spatial(features, positions, roles):
    features, positions, roles = np.asarray(features), np.asarray(positions), np.asarray(roles)
    fitting, testing = roles == "fitting", roles == "testing"
    if not fitting.any() or not testing.any() or len(features) != len(positions):
        raise ValueError("Spatial readout requires separate fitting and testing observations")
    model = make_pipeline(StandardScaler(), Ridge(alpha=1))
    model.fit(features[fitting], positions[fitting])
    prediction = model.predict(features[testing])
    return {"r2": float(model.score(features[testing], positions[testing])), "mse": float(np.square(prediction-positions[testing]).mean()), "n_testing": int(testing.sum()), "estimator": "StandardScaler+Ridge", "alpha": 1}
