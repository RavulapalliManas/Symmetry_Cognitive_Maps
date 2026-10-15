"""End-to-end artifact contract using an explicitly untrained real-cell fixture."""
import importlib
import json
import os
from pathlib import Path

import pytest
import torch


def test_replay_writes_traceable_artifacts_and_refuses_overwrite(tmp_path):
    assert importlib.util.find_spec("map_dynamics.replay") is not None, "legacy replay is missing"
    from map_dynamics.legacy import LegacyModules
    from map_dynamics.replay import run
    from map_dynamics.preparation import file_hash
    legacy = LegacyModules(Path(os.environ["MAP_DYNAMICS_LEGACY_ROOT"]))
    torch.manual_seed(3)
    meta = dict(obs_size=147, k=3, hidden_size=32, trunc=16, dropout_p=0., hidden_init_sigma=.1, pred_offset=0, condition="s2", hd_mode="axis", arena_size=18, noise_std=.03)
    model = legacy.architectures.pRNN_th(obs_size=147, act_size=5, k=3, hidden_size=32, trunc=16, dropp=0, cell=legacy.cells.LayerNormRNNCellEager)
    checkpoint = tmp_path / "untrained_fixture.pt"
    torch.save({"model": model.state_dict(), "meta": meta, "step": "untrained_fixture"}, checkpoint)
    config = {"legacy_root": str(legacy.root), "legacy_fingerprint": legacy.fingerprint(), "checkpoints": [{"path": str(checkpoint), "sha256": file_hash(checkpoint)}], "families": 6, "steps": 12, "seed": 100, "max_mem_gib": 2, "max_states": 1000, "threads": 2, "original_trajectory": None}
    output = tmp_path / "replay"
    report = run(config, output)
    assert report["status"] == "complete_reference_evaluation"
    assert report["scientific_hypotheses_supported"] is False
    assert report["models"][0]["exact_continuation_available"] is False
    assert (output / "metrics.csv").is_file()
    assert (output / "manifest.json").is_file()
    assert list((output / "bank").glob("*.agent.npz"))
    assert list((output / "population").rglob("*.npz"))
    assert json.loads((output / "report.json").read_text())["models"][0]["native_state_max_abs_error"] == 0
    with pytest.raises(FileExistsError):
        run(config, output)
