"""Forced-route X1 trials on the unchanged legacy C2 world.

Each family contains both spatial twins and both decision queries. All four
members stay in one split. Two reserved RGB pixels carry a binary cue and query;
the remaining pixels retain the legacy view. Motion contains realized speed,
axis heading, previous movement action, and previous reward. No analyst field
enters AgentInputs. A delay counts transitions from the last cue to commitment.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from scipy.sparse import eye

from map_dynamics.diagnostics import AgentInputs, BayesFilter


@dataclass(frozen=True)
class TrialMetadata:
    state_ids: np.ndarray
    family: np.ndarray
    context: np.ndarray
    query: np.ndarray
    target: np.ndarray


@dataclass(frozen=True)
class TrialBatch:
    agent: AgentInputs
    analyst: TrialMetadata


class TransientTask:
    """Finite, domain-restricted trials; feedback appears after commitment only."""
    def __init__(self, world, *, cue_duration=1, cue_reliability=.95,
                 start_state=(4, 4, 0), heading="axis", rule="context_specific_rule"):
        if type(cue_duration) is not int or cue_duration < 0:
            raise ValueError("cue_duration must be a nonnegative integer")
        if not np.isfinite(cue_reliability) or not 0 <= cue_reliability <= 1:
            raise ValueError("cue_reliability must be a probability")
        if heading not in {"axis", "const"}:
            raise ValueError("X1 requires a C2-invariant heading channel")
        if rule not in {"shared_rule", "context_specific_rule"}:
            raise ValueError("Unknown decision rule")
        self.world = world
        self.cue_duration, self.cue_reliability = cue_duration, cue_reliability
        self.start_state, self.heading, self.rule = tuple(start_state), heading, rule
        matches = np.flatnonzero(np.all(world.states == self.start_state, axis=1))
        if len(matches) != 1 or world.context[matches[0]] != 0:
            raise ValueError("start_state must be a passable state in the reference domain")
        self.start = int(matches[0])
        if not np.array_equal(world.views, world.views[world.twin]):
            raise ValueError("World observations violate C2 symmetry")
        if not np.array_equal(world.next[world.twin], world.twin[world.next]):
            raise ValueError("World transitions violate C2 symmetry")
        self.rng = np.random.default_rng()
        self.done = True

    def contract(self):
        return {"cue_duration": self.cue_duration, "cue_reliability": self.cue_reliability,
                "start_state": list(self.start_state), "heading": self.heading, "rule": self.rule,
                "input_version": "x1_reserved_rgb_pixels_arrival_motion_v1",
                "correct_reward": 1., "incorrect_reward": 0.,
                "delay_unit": "transitions_since_last_cue"}

    def batch(self, families, delay, seed):
        if type(families) is not int or families < 1 or type(delay) is not int or delay < 1:
            raise ValueError("families and delay must be positive integers")
        rng = np.random.default_rng(seed)
        frames = max(1, self.cue_duration) + delay
        states = np.empty((families * 4, frames), dtype=np.int64)
        signals = np.zeros((families * 4, frames, 10), dtype=np.float32)
        context = np.tile([0, 0, 1, 1], families)
        query = np.tile([0, 1, 0, 1], families)
        heading_codes = self.world.legacy.headings._matrix(self.heading).T
        symbols = np.empty(families * 4, dtype=np.int64)
        for family in range(families):
            first = family * 4
            path = np.empty(frames, dtype=np.int64)
            actions = np.full(frames, 3, dtype=np.int64)
            path[0] = self.start
            for time in range(1, frames):
                current = path[time-1]
                if time < self.cue_duration:
                    action = 3
                else:
                    allowed = self.world.context[self.world.next[current]] == 0
                    probabilities = self.world.action_probs * allowed
                    action = int(rng.choice(4, p=probabilities / probabilities.sum()))
                actions[time] = action
                path[time] = self.world.next[current, action]
            states[first:first+2] = path
            states[first+2:first+4] = self.world.twin[path]
            correct_cue = rng.random() < self.cue_reliability
            symbols[first:first+4] = context[first:first+4] ^ int(not correct_cue)
            for row in range(first, first+4):
                positions = self.world.states[states[row], :2]
                signals[row, 1:, 0] = np.linalg.norm(np.diff(positions, axis=0), axis=1)
                signals[row, :, 1:5] = heading_codes[self.world.states[states[row], 2]]
                signals[row, 1:, 5:9] = np.eye(4, dtype=np.float32)[actions[1:]]
        observations = self.world.views[states].copy()
        observations[:, :, :6] = 0  # Declared, permanent visual overlay locations.
        if self.cue_duration:
            observations[:, :self.cue_duration, 0] = symbols[:, None]
            observations[:, :self.cue_duration, 1] = 1-symbols[:, None]
            observations[:, :self.cue_duration, 2] = 1
        observations[:, -1, :3] = 1  # Common decision marker, carries no context.
        observations[:, -1, 3] = query
        observations[:, -1, 4] = 1-query
        observations[:, -1, 5] = 1
        target = query ^ context if self.rule == "context_specific_rule" else query.copy()
        return TrialBatch(AgentInputs(observations, signals), TrialMetadata(
            states, np.repeat(np.arange(families), 4), context, query, target))

    def context_beliefs(self, agent):
        """Exact context marginal for this paired, domain-restricted protocol.

        The initial prior assigns equal mass to the twins. Every other channel
        has equal likelihood under them. Repeated cue frames repeat one draw;
        they are not independent cue events. This is not a free-navigation filter.
        """
        if not isinstance(agent, AgentInputs):
            raise TypeError("Observer accepts permitted AgentInputs only")
        beliefs = np.empty((*agent.obs.shape[:2], 2), dtype=float)
        for row, observations in enumerate(agent.obs):
            observer = BayesFilter([eye(2, format="csr")], [.5, .5])
            if self.cue_duration:
                symbol = observations[0, :3]
                if tuple(symbol) not in {(0., 1., 1.), (1., 0., 1.)}:
                    raise ValueError("Unrecognized cue observation")
                likelihood = np.full(2, 1-self.cue_reliability)
                likelihood[int(symbol[0])] = self.cue_reliability
                observer.observe(likelihood)
            beliefs[row] = observer.belief
        return beliefs

    def reset(self, *, seed=None, delay=8):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.trial = self.batch(1, delay, int(self.rng.integers(2**31)))
        self.row = int(self.rng.integers(4))
        self.time, self.done = 0, False
        return self._observation(), self._metadata()

    def _observation(self):
        return AgentInputs(self.trial.agent.obs[self.row, self.time].copy(),
                           self.trial.agent.motion[self.row, self.time].copy())

    def _metadata(self):
        a = self.trial.analyst
        return {"state_id": int(a.state_ids[self.row, self.time]),
                "context": int(a.context[self.row]), "target": int(a.target[self.row]),
                "decision": self.time == self.trial.agent.obs.shape[1]-1}

    def step(self, action):
        if self.done:
            raise RuntimeError("Call reset before stepping a finished trial")
        if action not in (0, 1):
            raise ValueError("The response action must be 0 or 1")
        reward = 0.
        if self._metadata()["decision"]:
            reward = float(action == self.trial.analyst.target[self.row])
            self.done = True
        else:
            self.time += 1  # Movement is exogenous, not an actor-selected action.
        observation = self._observation()
        if self.done:
            observation.motion[0] = 0
            observation.motion[5:9] = np.eye(4, dtype=np.float32)[action]
            observation.motion[-1] = reward
        return observation, reward, self.done, False, self._metadata()

    def clone_state(self):
        return copy.deepcopy({"trial": self.trial, "row": self.row, "time": self.time,
                              "done": self.done, "rng": self.rng.bit_generator.state,
                              "contract": self.contract()})

    def restore_state(self, state):
        if state["contract"] != self.contract():
            raise ValueError("Task contract differs from snapshot")
        state = copy.deepcopy(state)
        self.trial, self.row, self.time, self.done = (state[k] for k in ("trial", "row", "time", "done"))
        self.rng.bit_generator.state = state["rng"]
