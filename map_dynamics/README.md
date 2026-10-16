# Learning when to share a cognitive map

This package prepares the computational study in the supplied design documents.
It implements a pinned legacy adapter, an exact C2 reference world, a sparse Bayesian observer, and held-out diagnostic readouts.
The X1 forced-route task and a bounded CPU development learner are implemented.
The scientific campaign and X2–X5 experiments remain unexecuted.

The maintained [perception, rearing, and policy design](docs/decisions/perception_rearing_policy.md)
records the discussion, biological interpretation, novelty limits, and ordered experiment amendments.
It retains direct sensory prediction and optional visual encoding within the same study.
Current amendments and source evidence live in `discussion_checkpoint` in
[preparation.yaml](registries/preparation.yaml); the supplied specification remains unchanged.

## Setup

Run installation and checks on `cpu-angad`. The Mac holds source files and copies them to the fleet.
From this package directory on the fleet:

```bash
uv sync --locked --extra science
MAP_DYNAMICS_LEGACY_ROOT="$PWD/legacy_reference" uv run --locked --extra science pytest -q tests
uv run --locked map-dynamics --root . validate
uv run --locked map-dynamics --root . needs
uv run --locked map-dynamics --root . prepare --experiment X1
```

The optional `science` dependencies pin the CPU environment tested on the fleet.
The source snapshot in `legacy_reference/` preserves the selected legacy implementation and its license.
Its provenance records the candidate checkout revision and file fingerprints.
The original training revision is absent from the selected checkpoints.

`prepare` prints a JSON draft. To save it, supply `--output` with a new path on the fleet.
The command refuses to overwrite an existing file.
The draft includes unresolved requirements and keeps `launchable` false.
It is not a frozen campaign or a run record.

```bash
uv run --locked map-dynamics --root . prepare --experiment X1 --output /YOUR/FLEET/OUTPUT/x1_draft.json
uv run --locked map-dynamics --root . audit-legacy --source-root /YOUR/LEGACY/CHECKOUT
```

`audit-legacy` inventories selected source files without importing the model or loading checkpoints.
It does not certify compatibility or reproduce a manuscript result.
The `replay-legacy` command produces a reference evaluation and source-linked artifact bundle.
The later intervention experiments remain unimplemented.

```bash
uv run --locked --extra science map-dynamics --root . replay-legacy \
  --config configs/protocols/X0_replay.yaml \
  --output /YOUR/FLEET/OUTPUT/reference_evaluation
```

The output path must be new. The command retains failed run status and refuses to overwrite earlier results.
It records complete recurrent states for the selected legacy cell and uses fitting data for decoder preprocessing.
The observer sees legacy SpeedHD inputs and marginalizes hidden turns.
It does not receive analyst action identities.
Policy use remains unavailable because these checkpoints have no trained policy head.

## Current status

The scaffold contains the supplied design, proposed experiment dependencies, configuration templates, and preparation tools.
The package runs selected legacy reference evaluations and bounded X1 software-development checks.
New scientific learning experiments remain unexecuted.
The first selected-checkpoint replay completed on the fleet.
The current implementation milestone is the transient-cue task and learning protocol for X1.
No campaign is authorized.

### X1 forced-route development

`transient.py` reuses the exact C2 world. Two reserved RGB pixels display the context cue and final binary query.
The overlay replaces those pixels throughout each trial; it is a new observation protocol, not legacy checkpoint input.
Matched twins remain within corresponding domains. Delay counts transitions from the last cue frame to the decision.
Repeated cue frames repeat one noisy symbol. The context observer counts this as one evidence event.
The shared rule reports the query. The context-specific rule reports query XOR context.
Correct responses earn one; incorrect responses earn zero. Feedback appears after commitment.

`learning.py` reuses the unchanged legacy pRNN cell and predictive rollout with the extended input.
It adds policy/value heads and predictive, task, joint, detached-actor, and frozen-population loss routing.
PPO applies only to the final response. Every optimization epoch replays the full episode from its saved initial state and noise.
The new protocol uses Adam, no observation dropout, and an explicit combined update. It does not reproduce the historical training schedule.
Checkpoints contain model, optimizer, random states, counters, protocol, and implementation fingerprints.
Diagnostics use fixed paired histories and family splits; fitting data alone determine decoder preprocessing.
Population files contain the complete hidden trajectory, initial state, and noise.

Run the development command on the fleet inside its managed runner, with an external memory cap:

```bash
systemd-run --user --scope -p MemoryMax=2G -p CPUQuota=200% \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  uv run --locked --extra science map-dynamics --root . develop-x1 \
  --config configs/protocols/X1_development.yaml --output /YOUR/NEW/FLEET/OUTPUT
```

The command refuses existing output directories. `--stop-after` stops at a total rollout-update count.
Use `--resume /PATH/checkpoint.pt` and a new output directory to continue the identical configured protocol.
Caps restrict this command to short CPU development. They do not authorize the registered pilot.
The full-scale learner, room-report capacity control, fixed-readout transfer, and final campaign protocol remain pending.
Development choices and verification records live in `implementation.X1` in [preparation.yaml](registries/preparation.yaml).

The X1 draft uses the registered task and learning-mode combinations.
Evaluation delays remain evaluation settings and do not multiply the training arms.
Development seed identifiers are proposed preparation defaults, not frozen campaign decisions.
Positive controls require separate implementation and budgeting.

The source inventory records the current parent checkout as a candidate.
It does not declare that the current revision matches the submitted manuscript or original checkpoints.
Existing source files and historical results remain in the parent repository.

### Cue perception development

X1 also supports `identity`, `frozen_cnn`, and `trainable_cnn` visual pathways.
The CNN processes frames independently before the unchanged recurrent cell. It has no temporal state or pretrained weights.
The frozen CNN is a random-feature control. All pathways predict the same raw RGB targets.
Encoder initialization preserves the paired backbone, policy/value heads, and subsequent random draws.

`cue_contrast` changes the rendered cue separately from `cue_reliability`, which controls its truthfulness.
Zero contrast removes symbol identity. Nonzero representable contrast leaves the exact observer's validity-based posterior unchanged.
This is a rendering-sensitivity assay, not a calibrated perceptual threshold or a natural landmark task.
The observer rejects positive contrast that collapses under float32 rendering.

Diagnostics fit raw-cue, encoder-cue, post-cue recurrent, and delayed recurrent readouts on fitting families only.
The presented symbol and true context remain separate analyst targets. Neither enters the model as a label.
Population files include visual features as well as the complete recurrent trajectory, initial state, and noise.
The extended checkpoint contract records the encoder and contrast. Changed source/protocol rejects incompatible continuation.

Use the existing managed CPU command with `configs/protocols/X1_perception_development.yaml`.
Its limits remain development limits. The declared encoder-by-contrast runs and their status are recorded in
`discussion_checkpoint.development` in [preparation.yaml](registries/preparation.yaml).
An immediate-response capacity control, scene-embedded cues, rearing branches, and state-driven free navigation remain pending.

### Biological datasets, novelty, and learning algorithms

On 2026-10-16, six Luna reviews examined biological datasets, prior work, learning algorithms, and optimality.
The source-linked shortlist and unresolved access checks are in `research_review` in [preparation.yaml](registries/preparation.yaml).
No dataset was downloaded or analyzed during this review.

The preferred neural-learning anchor is the [Sun cue-delay-choice dataset](https://doi.org/10.25378/janelia.27273552).
The [de Cothi data and code](https://zenodo.org/records/6734981) provide a rat-human behavioral comparison.
Odor memory, delivered cue updates, fish avoidance, and fly orientation provide narrower complementary comparisons.
These tasks do not jointly establish a cross-species neural learning mechanism.

[Sun et al.](https://doi.org/10.1038/s41586-024-08548-w) already compare map development with recurrent, LSTM, Transformer, and latent-state models.
The proposed contribution therefore needs controlled evidence access, memory or policy interventions, and prospective predictions beyond model rankings.
Learning objectives, credit assignment, architecture, and sensory access remain separate experimental factors.
Bayes context accuracy, reward-optimal control, bounded-memory performance, and learning efficiency require different benchmarks.
All extensions remain proposed; the review does not change hypothesis status or authorize a campaign.

## Inputs needed before scientific work

Run `map-dynamics --root . needs` for the maintained requirements in [preparation.yaml](registries/preparation.yaml).

- For M0, locate a checkpoint with its configuration, trajectories, and the source manuscript.
- For M0, identify and test its source revision, input semantics, and heading aliases.
- For M1, implement the observer, split isolation, complete-state recording, and source-linked analysis.
- For M2, resolve the task timing, training delays, rewards, optimizer schedule, splits, and stopping budgets.
- Before launch, obtain hardware, compute limits, output paths, durable storage, and campaign authorization.

The historical external-drive locations were unavailable during initial inspection.
That observation does not establish that the assets are absent from the fleet or other storage.

## Design and provenance

| File | Role |
|---|---|
| [RESEARCH_SPEC.md](docs/RESEARCH_SPEC.md) | Scientific questions and inference boundaries |
| [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) | Milestones, interfaces, data contracts, and acceptance criteria |
| [SOURCES.md](docs/SOURCES.md) | Supplied source ledger; inherited claims remain unaudited |
| [experiments.yaml](registries/experiments.yaml) | Original X0–X5 dependencies and proposed protocols |
| [claims.yaml](registries/claims.yaml) | Original hypotheses, all untested |
| [preparation.yaml](registries/preparation.yaml) | Current unresolved preparation and launch requirements |
| [assets.csv](registries/assets.csv) | Selected source inventory; no model compatibility claim |
| [PACKAGE_MANIFEST.json](PACKAGE_MANIFEST.json) | Input locations and checksums after migration |

The original package [README](docs/DESIGN_README.md), [agent instructions](docs/DESIGN_AGENTS.md),
and [manifest](docs/PACKAGE_MANIFEST.json) remain unchanged as supplied records.
The [combined document](docs/map_dynamics_research_and_repo_spec.md) also remains unchanged.
Historical links inside those supplied documents use their original package layout.
The table above gives the maintained paths in this checkout.

Configuration templates live under `configs/`.
Preparation tools and the implemented reference pipeline live under `src/map_dynamics/`.
Later scientific modules will be added with their real implementation and focused checks.
The [contributor instructions](AGENTS.md) govern changes to this package.

Setup follows the [uv project layout](https://docs.astral.sh/uv/concepts/projects/init/)
and [locking workflow](https://docs.astral.sh/uv/concepts/projects/sync/).
Preparation parses configuration with [PyYAML safe loading](https://pyyaml.org/wiki/PyYAMLDocumentation).

## Verification record

On 2026-10-15, installation and preparation checks ran on `cpu-angad` with Python 3.12.
`uv run --locked pytest -q tests` returned `9 passed` and exit code 0.
`uv run --locked map-dynamics --root . validate` returned `preparation_consistent` and exit code 0.
The validator kept scientific execution unverified and launchability false.

Source on the fleet: `~/symm/map_dynamics/`.
Initial draft plans: `~/symm/results/map_dynamics/setup/prepared/`.
Verification logs: `~/symm/results/map_dynamics/setup/`.
The fleet run registry records setup under `map-dynamics-setup`.

The initial install failed with exit code 134 when the worker thread pool could not initialize under the memory cap.
Restricting worker threads resolved installation without changing the cap.
The initial preparation tests failed because the API and CLI were not yet implemented.
The logs preserve those failures in `install.log` and `tests_before.log`.

These results establish preparation behavior. The later learning and causal acceptance criteria remain unexecuted.

### Reference pipeline milestone

On 2026-10-15, the focused suite returned `23 passed` and exit code 0 on cpu-angad.
The suite reports one deprecation warning from the preserved legacy TorchScript code.
The check used `MAP_DYNAMICS_LEGACY_ROOT=.../legacy_reference .venv/bin/python -m pytest -q tests`.
Scientific tests include exact input identity, transition equivariance, known observer posteriors, family splits, decoder nulls, weak-signal detection, and fixed-versus-refitted readouts.

The real command `map-dynamics --root . replay-legacy --config configs/protocols/X0_replay.yaml --output ~/symm/results/map_dynamics/x0/reference_01` returned exit code 0.
The bundle contains `report.json`, `metrics.csv`, `manifest.json`, `splits.json`, input banks, and complete-state recordings.
Logs live in `~/symm/results/map_dynamics/x0/`.

The selected checkpoints are weight-only. They cannot reproduce exact optimizer or RNG continuation.
The audit verifies evaluation compatibility with the pinned source, not identity with the original training environment.
The manuscript fingerprint and `flip` alias remain unreconciled.
No cloud archive of this new bundle is verified.
The X1 pilot and all causal or transfer claims remain untested.

### X1 software-development milestone

On 2026-10-16, `MAP_DYNAMICS_LEGACY_ROOT=.../legacy_reference uv run --locked --extra science pytest -q tests`
returned `41 passed` and exit code 0 on cpu-angad. The inherited TorchScript deprecation warning remains.
The tests cover paired inputs, cue posteriors, response timing, gradient routing, prediction isolation, and exact checkpoint continuation.
The capped `develop-x1` command returned `complete_development_run` and exit code 0.
The artifact directory contains checkpoints, fixed diagnostic banks, population recordings, metrics, and source provenance.
Independent review identified a runtime flag missing from the continuation contract. The regression now rejects a changed execution path.
The final development artifact directory is `smoke_reviewed/`; the initial run and failed regression logs remain preserved.
Logs and outputs live under `cpu-angad:~/symm/results/map_dynamics/x1_development_20261016/`.
Preparation validation returned `preparation_consistent` with launchability false.
These checks establish execution and continuation. They do not establish learning quality or support a study hypothesis.

### Perception software-development milestone

On 2026-10-17, the focused command `MAP_DYNAMICS_LEGACY_ROOT=legacy_reference uv run --locked --extra science pytest -q tests/test_x1.py`
returned `24 passed` and exit code 0 on cpu-angad. The inherited TorchScript warning remains.
The tests include cue-contrast rendering, known positive/null readouts, framewise encoding, gradient routing, paired initialization, and exact continuation.
The preserved red run returned six expected failures before implementation.

The sequential development matrix completed all nine declared encoder-by-contrast arms, each with exit code 0.
Each arm uses the same seed and four updates, within the existing CPU development caps.
These are execution checks, not independent scientific replicates or evidence of learning quality.
The matrix report and per-arm checkpoints, diagnostics, banks, and population files live at
`cpu-angad:~/symm/results/map_dynamics/perception_development_20261017/`.
Completed artifacts are archived at `gs://perseus-swe-rl/scratch/map_dynamics/perception_development_20261017/`.
The source/config snapshots, failed tests, final tests, and matrix driver are retained with those artifacts.
The preparation validator returned `preparation_consistent` with exit code 0; scientific launchability remains false.
