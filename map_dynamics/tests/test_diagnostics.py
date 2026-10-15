"""Information and split invariants with independently specified answers."""
import importlib
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix, eye


def module():
    assert importlib.util.find_spec("map_dynamics.diagnostics") is not None, "diagnostics are missing"
    return importlib.import_module("map_dynamics.diagnostics")


def test_filter_no_evidence_and_known_binary_cue_posterior():
    m = module()
    observer = m.BayesFilter([eye(2, format="csr")], [.5, .5])
    np.testing.assert_allclose(observer.observe([1, 1]), [.5, .5])
    np.testing.assert_allclose(observer.observe([.95, .05]), [.95, .05])
    observer.predict(0)
    np.testing.assert_allclose(observer.observe([1, 1]), [.95, .05])
    np.testing.assert_allclose(observer.observe([0, 1]), [0, 1])


def test_filter_stochastic_transition_and_impossible_observation():
    m = module()
    observer = m.BayesFilter([csr_matrix([[.8, .2], [.1, .9]])], [1, 0])
    np.testing.assert_allclose(observer.predict(0), [.8, .2])
    with pytest.raises(ValueError, match="impossible"):
        observer.observe([0, 0])
    np.testing.assert_allclose(observer.belief, [.8, .2])
    with pytest.raises(ValueError, match="stochastic"):
        m.BayesFilter([csr_matrix([[1, 1], [0, 1]])], [.5, .5])


def test_family_split_keeps_all_twins_and_timesteps_together():
    split = module().family_split(np.repeat(np.arange(12), 24), seed=7)
    assert set(split) == {"fitting", "validation", "testing"}
    for family in range(12):
        assert len(set(split[np.repeat(np.arange(12), 24) == family])) == 1


def test_exact_c2_renderer_and_transitions_respect_half_turn():
    from map_dynamics.legacy import LegacyModules
    legacy = LegacyModules(Path(os.environ["MAP_DYNAMICS_LEGACY_ROOT"]))
    world = module().C2World(legacy)
    np.testing.assert_array_equal(world.views, world.views[world.twin])
    np.testing.assert_array_equal(world.next[world.twin], world.twin[world.next])
    assert np.all(world.context != world.context[world.twin])
    family = world.paired_trajectory(steps=24, seed=3, hd_mode="axis")
    np.testing.assert_array_equal(family.agent.obs[0], family.agent.obs[1])
    np.testing.assert_array_equal(family.agent.motion[0], family.agent.motion[1])
    assert not set(family.agent.__dataclass_fields__) & {"context", "position", "state_id"}
    accuracy = world.observer_accuracy(family.agent.obs[0], family.agent.motion[0], "axis")
    np.testing.assert_allclose(accuracy, .5)


def test_decoder_detects_context_and_rejects_constant_and_path_id():
    m = module()
    family = np.repeat(np.arange(24), 16)
    context = np.tile([0, 1], len(family) // 2)
    roles = m.family_split(family, seed=7)
    label_code = (2 * context - 1).astype(float)[:, None]
    assert m.decode(label_code, context, roles)["accuracy"] == 1
    assert m.decode(np.zeros_like(label_code), context, roles)["accuracy"] == .5
    path_code = np.eye(24)[family]
    assert m.decode(path_code, context, roles)["accuracy"] == .5
    policy_choices = np.zeros(len(context), dtype=int)
    assert np.mean(policy_choices == context) == .5


def test_fixed_decoder_distinguishes_recoding_from_loss():
    m = module()
    family = np.repeat(np.arange(24), 16)
    y = np.tile([0, 1], len(family) // 2)
    roles = m.family_split(family, seed=4)
    early = np.column_stack([2*y-1, np.zeros(len(y))])
    rotated = np.column_stack([np.zeros(len(y)), 2*y-1])
    fixed = m.decode(early, y, roles, evaluation_features=rotated)
    refit = m.decode(rotated, y, roles)
    assert fixed["accuracy"] == .5
    assert refit["accuracy"] == 1
    assert m.decode(np.zeros_like(rotated), y, roles)["accuracy"] == .5


def test_nonlinear_readout_recovers_context_and_rejects_constant_code():
    m = module()
    family = np.repeat(np.arange(24), 16)
    y = np.tile([0, 1], len(family) // 2)
    roles = m.family_split(family, seed=4)
    assert m.decode((2*y-1)[:, None], y, roles, kind="knn")["accuracy"] == 1
    assert m.decode(np.zeros((len(y), 1)), y, roles, kind="knn")["accuracy"] == .5


def test_weak_signal_is_detected_with_prespecified_seed_and_effect():
    m = module()
    family = np.repeat(np.arange(24), 128)
    y = np.tile([0, 1], len(family) // 2)
    roles = m.family_split(family, seed=4)
    weak = .25*(2*y-1)[:, None] + np.random.default_rng(109).normal(size=(len(y), 1))
    assert .55 < m.decode(weak, y, roles)["accuracy"] < .7


def test_spatial_readout_recovers_known_coordinates_and_fails_without_signal():
    m = module()
    family = np.repeat(np.arange(24), 16)
    roles = m.family_split(family, seed=4)
    positions = np.random.default_rng(11).normal(size=(len(family), 2))
    assert m.decode_spatial(positions, positions, roles)["r2"] > .99
    assert m.decode_spatial(np.zeros_like(positions), positions, roles)["r2"] <= .01
