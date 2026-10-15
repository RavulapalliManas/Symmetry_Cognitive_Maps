# Learning when to share a cognitive map

This package prepares the computational study in the supplied design documents.
It implements a pinned legacy adapter, an exact C2 reference world, a sparse Bayesian observer, and held-out diagnostic readouts.
The X1–X5 learning experiments and their training runner remain unimplemented.

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
Training commands and the later intervention experiments remain unimplemented.

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
The package runs selected legacy reference evaluations. New learning experiments remain unexecuted.
The first selected-checkpoint replay completed on the fleet.
The next implementation milestone is the transient-cue task and learning protocol for X1.
No campaign is authorized.

The X1 draft uses the registered task and learning-mode combinations.
Evaluation delays remain evaluation settings and do not multiply the training arms.
Development seed identifiers are proposed preparation defaults, not frozen campaign decisions.
Positive controls require separate implementation and budgeting.

The source inventory records the current parent checkout as a candidate.
It does not declare that the current revision matches the submitted manuscript or original checkpoints.
Existing source files and historical results remain in the parent repository.

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
