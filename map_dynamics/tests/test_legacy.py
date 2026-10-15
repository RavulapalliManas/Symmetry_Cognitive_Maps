"""Real-cell compatibility tests. No pretrained fixture is fabricated."""
import importlib
import os
from pathlib import Path

import pytest
import torch


def module():
    assert importlib.util.find_spec("map_dynamics.legacy") is not None, "legacy adapter is missing"
    return importlib.import_module("map_dynamics.legacy")


@pytest.fixture
def legacy():
    return Path(os.environ["MAP_DYNAMICS_LEGACY_ROOT"])


def test_online_cell_sequence_prediction_and_rng_match_native(legacy):
    legacy = module().LegacyModules(legacy)
    torch.manual_seed(10)
    model = legacy.architectures.pRNN_th(
        obs_size=147, act_size=5, hidden_size=32, k=3, trunc=8,
        cell=legacy.cells.LayerNormRNNCellEager, dropp=0,
    ).eval()
    adapter = module().LegacyAdapter(model)
    obs, act = torch.rand(2, 18, 147), torch.rand(2, 17, 5)
    initial = torch.rand(2, 32) * .1
    noise = torch.randn(2, 17, 32) * .03
    original_initial = initial.clone()
    native = model._forward_main_trajectory(obs[:, :-1], act, state=initial.unsqueeze(0), noise_main=noise)
    actual = adapter.record(obs[:, :-1], act, initial, noise)
    torch.testing.assert_close(actual, native, atol=0, rtol=0)
    native_pred, native_h, native_target = model(obs, act, state=initial.unsqueeze(0), noise_main=noise)
    pred, hidden, target = adapter.predict(obs, act, initial, noise)
    for a, b in zip((pred, hidden, target), (native_pred, native_h, native_target)):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(initial, original_initial, atol=0, rtol=0)
    torch.testing.assert_close(hidden, actual[:, :14], atol=0, rtol=0)


def test_missing_architecture_metadata_fails_instead_of_guessing(legacy):
    legacy = module().LegacyModules(legacy)
    with pytest.raises(ValueError, match="metadata"):
        legacy.load_checkpoint({"meta": {}, "model": {}})


def test_source_fingerprint_detects_modified_source(legacy, tmp_path):
    legacy = module().LegacyModules(legacy)
    pinned = legacy.fingerprint()
    assert pinned["utils/Architectures.py"]
    pinned["utils/Architectures.py"] = "bad"
    with pytest.raises(ValueError, match="fingerprint"):
        legacy.assert_fingerprint(pinned)


def test_heading_alias_maps_are_not_assumed(legacy):
    legacy = module().LegacyModules(legacy)
    report = legacy.heading_contract()
    assert report["axis_half_turn_invariant"] is True
    assert report["parity_half_turn_invariant"] is False
    assert report["flip_matrix_verified"] is False
