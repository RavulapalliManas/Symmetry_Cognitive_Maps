"""X1 task, information, gradient-routing, and continuation contracts."""
import importlib
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def api(name):
    assert importlib.util.find_spec(f"map_dynamics.{name}") is not None, f"X1 {name} is missing"
    return importlib.import_module(f"map_dynamics.{name}")


@pytest.fixture(scope="module")
def world():
    from map_dynamics.diagnostics import C2World
    from map_dynamics.legacy import LegacyModules
    return C2World(LegacyModules(Path(os.environ["MAP_DYNAMICS_LEGACY_ROOT"])))


def task(world, **kwargs):
    return api("transient").TransientTask(world, **kwargs)


def test_cue_twins_share_every_delayed_input_and_remain_in_their_domains(world):
    batch = task(world).batch(families=6, delay=8, seed=17)
    assert batch.agent.obs.shape == (24, 9, 147)
    assert batch.agent.motion.shape == (24, 9, 10)
    for start in range(0, 24, 4):
        for query in (0, 1):
            a, b = start + query, start + 2 + query
            np.testing.assert_array_equal(batch.agent.obs[a, 1:], batch.agent.obs[b, 1:])
            np.testing.assert_array_equal(batch.agent.motion[a], batch.agent.motion[b])
            assert np.all(world.context[batch.analyst.state_ids[a]] == 0)
            assert np.all(world.context[batch.analyst.state_ids[b]] == 1)
    assert not set(batch.agent.__dataclass_fields__) & {"context", "target", "family"}
    assert np.all(batch.agent.motion[:, :, -1] == 0), "No predecision reward"
    np.testing.assert_array_equal(batch.agent.motion[:, 0, 5:9], 0)


def test_reliability_observer_counts_one_cue_event_not_repeated_frames(world):
    env = task(world, cue_duration=3, cue_reliability=.75)
    batch = env.batch(families=2, delay=8, seed=1)
    beliefs = env.context_beliefs(batch.agent)
    symbols = batch.agent.obs[:, 0, 0].astype(int)
    for row, symbol in enumerate(symbols):
        assert np.allclose(beliefs[row, :, symbol], .75)
        assert np.allclose(beliefs[row, :, 1-symbol], .25)
    null = task(world, cue_duration=0)
    assert np.allclose(null.context_beliefs(null.batch(2, 8, 1).agent), .5)
    for reliability in (0., .5, 1.):
        env = task(world, cue_reliability=reliability)
        b = env.batch(2, 8, 1)
        assert np.allclose(env.context_beliefs(b.agent).max(-1), max(reliability, 1-reliability))


def test_query_and_rule_change_target_without_changing_experience(world):
    shared = task(world, rule="shared_rule").batch(3, 8, 4)
    specific = task(world, rule="context_specific_rule").batch(3, 8, 4)
    np.testing.assert_array_equal(shared.agent.obs, specific.agent.obs)
    np.testing.assert_array_equal(shared.agent.motion, specific.agent.motion)
    np.testing.assert_array_equal(shared.analyst.target, shared.analyst.query)
    np.testing.assert_array_equal(specific.analyst.target, specific.analyst.query ^ specific.analyst.context)
    assert np.mean(specific.analyst.target == shared.analyst.query) == .5


def test_step_feedback_timing_and_restore(world):
    env = task(world)
    observation, info = env.reset(seed=4, delay=8)
    snapshot = env.clone_state()
    first = env.step(0)
    env.restore_state(snapshot)
    second = env.step(1)
    np.testing.assert_array_equal(first[0].obs, second[0].obs)
    assert first[1:4] == second[1:4] == (0., False, False)
    while not second[4]["decision"]:
        second = env.step(0)
    correct = second[4]["target"]
    terminal = env.step(correct)
    assert terminal[1:4] == (1., True, False)
    assert terminal[0].motion[-1] == 1
    assert terminal[0].motion[5+correct] == 1
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)


def trainer(world, mode="joint"):
    return api("learning").ForcedRouteLearner(
        world.legacy, task(world), mode=mode, hidden_size=16,
        prediction_horizon=2, seed=9, noise_std=.03,
    )


@pytest.mark.parametrize("mode", ["predictive_only", "task_only", "joint", "predictive_stop_gradient_actor", "frozen_population"])
def test_loss_gradients_reach_only_declared_parameters(world, mode):
    learner = trainer(world, mode)
    rollout = learner.collect(families=2, delay=8)
    losses = learner.losses(rollout)
    learner.optimizer.zero_grad(set_to_none=True)
    losses["total"].backward()
    groups = {
        "core": learner.model.backbone.rnn.cell.parameters(),
        "prediction": learner.model.backbone.outlayer.parameters(),
        "actor": learner.model.actor.parameters(),
        "critic": learner.model.critic.parameters(),
    }
    active = {name: any(p.grad is not None and torch.any(p.grad != 0) for p in ps) for name, ps in groups.items()}
    assert active["core"] == (mode != "frozen_population")
    assert active["prediction"] == (mode in {"predictive_only", "joint", "predictive_stop_gradient_actor"})
    assert active["actor"] == (mode != "predictive_only")
    assert active["critic"] == (mode != "predictive_only")
    if mode == "predictive_stop_gradient_actor":
        learner.optimizer.zero_grad(set_to_none=True)
        losses = learner.losses(rollout)
        (losses["actor"] + losses["critic"] + losses["entropy"]).backward()
        assert all(p.grad is None for p in learner.model.backbone.parameters())


def test_prediction_matches_native_and_does_not_modify_live_state(world):
    learner = trainer(world)
    rollout = learner.collect(2, 8)
    obs, signals = rollout["obs"], rollout["signals"]
    activity = learner.model.activity(obs, signals, rollout["initial"], rollout["noise"])
    before = activity.detach().clone()
    prediction, target = learner.model.predict(activity, obs, signals)
    native, _, expected = learner.model.backbone(obs, signals, state=rollout["initial"], noise_main=rollout["noise"])
    torch.testing.assert_close(prediction, native, rtol=0, atol=0)
    torch.testing.assert_close(target, expected, rtol=0, atol=0)
    torch.testing.assert_close(activity, before, rtol=0, atol=0)
    changed = signals.clone()
    changed[:, 5:] += 2
    prefix = learner.model.activity(obs, changed, rollout["initial"], rollout["noise"])
    torch.testing.assert_close(prefix[:, :5], before[:, :5], rtol=0, atol=0)
    # Full episode replay uses the saved initial state and noise under unchanged weights.
    logits = learner.model.actor(activity[:, -1])
    logs = torch.distributions.Categorical(logits=logits).log_prob(rollout["actions"])
    torch.testing.assert_close(logs, rollout["old_log_probs"], rtol=0, atol=0)


def test_checkpoint_continuation_restores_optimizer_data_and_torch_rng(world, tmp_path):
    a = trainer(world)
    a.update(2, 8)
    checkpoint = tmp_path / "checkpoint.pt"
    a.save(checkpoint)
    expected = a.update(2, 8)
    parameters = {k: v.detach().clone() for k, v in a.model.state_dict().items()}
    b = trainer(world)
    b.load(checkpoint)
    actual = b.update(2, 8)
    assert expected == actual
    for key, value in b.model.state_dict().items():
        torch.testing.assert_close(value, parameters[key], rtol=0, atol=0)
    assert b.updates == 2
    assert b.optimizer_steps == 4
    with pytest.raises(ValueError, match="contract"):
        trainer(world, "task_only").load(checkpoint)


def test_diagnostic_evaluation_does_not_change_training_rng(world, tmp_path):
    learner = trainer(world)
    batch = learner.task.batch(6, 8, 1701)
    from map_dynamics.diagnostics import family_split
    before_torch = torch.get_rng_state().clone()
    before_data = learner.rng.bit_generator.state
    api("learning").evaluate(learner, batch, family_split(batch.analyst.family, 2), tmp_path / "states.npz")
    torch.testing.assert_close(torch.get_rng_state(), before_torch, rtol=0, atol=0)
    assert learner.rng.bit_generator.state == before_data


@pytest.mark.parametrize("change", [{"prediction_weight": -1.}, {"learning_rate": float("nan")}, {"train_delays": []}])
def test_runner_rejects_invalid_objectives_and_empty_training_schedule(change, tmp_path):
    config = yaml.safe_load((ROOT / "configs/protocols/X1_development.yaml").read_text())
    config.update(change)
    with pytest.raises(ValueError, match="development"):
        api("learning").run_development(ROOT, config, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_resume_rejects_changed_experience_protocol(world, tmp_path):
    a = trainer(world)
    a.protocol = {"train_delays": [8]}
    a.save(tmp_path / "checkpoint.pt")
    b = trainer(world)
    b.protocol = {"train_delays": [16]}
    with pytest.raises(ValueError, match="protocol"):
        b.load(tmp_path / "checkpoint.pt")


def test_resume_rejects_changed_legacy_gradient_execution_path(world, tmp_path, monkeypatch):
    monkeypatch.setattr(world.legacy.architectures, "_HOIST_INPUT_PROJ", False)
    a = trainer(world)
    a.save(tmp_path / "checkpoint.pt")
    monkeypatch.setattr(world.legacy.architectures, "_HOIST_INPUT_PROJ", True)
    with pytest.raises(ValueError, match="contract"):
        trainer(world).load(tmp_path / "checkpoint.pt")


def test_runner_is_bounded_records_diagnostics_and_refuses_overwrite(world, tmp_path):
    m = api("learning")
    config = yaml.safe_load((ROOT / "configs/protocols/X1_development.yaml").read_text())
    config.update(updates=2, hidden_size=16, batch_families=2, prediction_horizon=2,
                  evaluation_families=6, evaluation_delays=[4, 8], train_delays=[4, 8], checkpoint_every=1)
    out = tmp_path / "run"
    result = m.run_development(ROOT, config, out)
    assert result["status"] == "complete_development_run"
    assert result["scientific_hypotheses_supported"] is False
    assert (out / "checkpoint_000002.pt").is_file()
    assert (out / "metrics.jsonl").read_text().count("\n") == 2
    assert len(list((out / "population").glob("*.npz"))) == 6
    with pytest.raises(FileExistsError):
        m.run_development(ROOT, config, out)
    config["updates"] = 1000000
    with pytest.raises(ValueError, match="development"):
        m.run_development(ROOT, config, tmp_path / "too_large")


def test_cue_contrast_changes_visibility_without_changing_validity_or_routes(world):
    clear = task(world, cue_reliability=.75, cue_contrast=1.).batch(6, 8, 17)
    weak_task = task(world, cue_reliability=.75, cue_contrast=.25)
    weak = weak_task.batch(6, 8, 17)
    np.testing.assert_array_equal(clear.analyst.cue_symbol, weak.analyst.cue_symbol)
    np.testing.assert_array_equal(clear.analyst.state_ids, weak.analyst.state_ids)
    np.testing.assert_array_equal(clear.agent.motion, weak.agent.motion)
    np.testing.assert_array_equal(clear.agent.obs[:, 1:], weak.agent.obs[:, 1:])
    np.testing.assert_allclose(weak.agent.obs[:, 0, 0], .5 + .25*(weak.analyst.cue_symbol-.5))
    assert np.allclose(weak_task.context_beliefs(weak.agent).max(-1), .75)
    invisible = task(world, cue_reliability=1., cue_contrast=0.)
    assert np.allclose(invisible.context_beliefs(invisible.batch(6, 8, 17).agent), .5)
    for contrast in (-1., 2., float("nan"), 1e-20):
        with pytest.raises(ValueError, match="contrast"):
            task(world, cue_contrast=contrast)


def test_encoder_is_framewise_and_preserves_paired_core_initialization(world):
    m = api("learning")
    torch.manual_seed(9)
    raw = m.ActiveModel(world.legacy, 16, 2)
    after_raw = torch.get_rng_state().clone()
    torch.manual_seed(9)
    encoded = m.ActiveModel(world.legacy, 16, 2, visual_encoder="trainable_cnn")
    torch.testing.assert_close(torch.get_rng_state(), after_raw, rtol=0, atol=0)
    for name, value in raw.state_dict().items():
        torch.testing.assert_close(encoded.state_dict()[name], value, rtol=0, atol=0)
    obs = torch.from_numpy(task(world).batch(2, 8, 17).agent.obs)
    torch.testing.assert_close(raw.visual_features(obs), obs, rtol=0, atol=0)
    before = encoded.visual_features(obs)
    changed = obs.clone()
    changed[:, 1:] = 0
    torch.testing.assert_close(encoded.visual_features(changed)[:, 0], before[:, 0], rtol=0, atol=0)
    assert before.shape == obs.shape
    with pytest.raises(ValueError, match="encoder"):
        m.ActiveModel(world.legacy, 16, 2, visual_encoder="unknown")


@pytest.mark.parametrize("encoder", ["frozen_cnn", "trainable_cnn"])
def test_encoder_gradient_routing_and_raw_prediction_targets(world, encoder):
    learner = api("learning").ForcedRouteLearner(
        world.legacy, task(world), hidden_size=16, prediction_horizon=2,
        visual_encoder=encoder,
    )
    rollout = learner.collect(2, 8)
    activity = learner.model.activity(rollout["obs"], rollout["signals"], rollout["initial"], rollout["noise"])
    predicted, target = learner.model.predict(activity, rollout["obs"], rollout["signals"])
    encoded = learner.model.visual_features(rollout["obs"])
    native, _, _ = learner.model.backbone(
        encoded, rollout["signals"], state=rollout["initial"], noise_main=rollout["noise"])
    torch.testing.assert_close(predicted, native, rtol=0, atol=0)
    torch.testing.assert_close(target[:, 0], rollout["obs"][:, :-2], rtol=0, atol=0)
    learner.losses(rollout)["total"].backward()
    gradients = [p.grad for p in learner.model.visual_encoder.parameters()]
    assert any(g is not None and torch.any(g != 0) for g in gradients) == (encoder == "trainable_cnn")


def test_perception_diagnostics_distinguish_visible_symbol_from_hidden_context(world, tmp_path):
    m = api("learning")
    env = task(world, cue_contrast=0., cue_reliability=1.)
    learner = m.ForcedRouteLearner(world.legacy, env, hidden_size=16, prediction_horizon=2)
    batch = env.batch(12, 8, 17)
    from map_dynamics.diagnostics import family_split
    path = tmp_path / "states.npz"
    result = m.evaluate(learner, batch, family_split(batch.analyst.family, 2), path)
    assert result["cue_symbol_linear"]["accuracy"] == .5
    assert result["cue_context_linear"]["accuracy"] == .5
    assert result["post_cue_context_linear"]["accuracy"] == .5
    assert result["bayes_context_accuracy"] == .5
    with np.load(path) as saved:
        assert saved["visual_features"].shape == batch.agent.obs.shape
        assert saved["cue_frame"] == 0
    assert all(p.grad is None for p in learner.model.parameters())
    visible = task(world, cue_contrast=.25, cue_reliability=1.)
    positive = visible.batch(12, 8, 17)
    learner.task = visible
    result = m.evaluate(learner, positive, family_split(positive.analyst.family, 2), tmp_path / "visible.npz")
    assert result["raw_cue_symbol_linear"]["accuracy"] == 1.
    assert result["cue_symbol_linear"]["accuracy"] == 1.
    assert result["cue_context_linear"]["accuracy"] == 1.


def test_encoder_checkpoint_continuation_restores_exact_updates(world, tmp_path):
    m = api("learning")
    kwargs = dict(hidden_size=16, prediction_horizon=2, visual_encoder="trainable_cnn")
    first = m.ForcedRouteLearner(world.legacy, task(world), **kwargs)
    first.update(2, 8)
    path = tmp_path / "encoder.pt"
    first.save(path)
    expected = first.update(2, 8)
    second = m.ForcedRouteLearner(world.legacy, task(world), **kwargs)
    second.load(path)
    assert second.update(2, 8) == expected
    for name, value in first.model.state_dict().items():
        torch.testing.assert_close(second.model.state_dict()[name], value, rtol=0, atol=0)
