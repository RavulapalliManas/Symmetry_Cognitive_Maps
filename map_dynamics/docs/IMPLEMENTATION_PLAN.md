# Implementation plan

**Status:** Design only. The package does not contain an implemented environment, model, trainer, experiment runner, or executed scientific test suite. Proposed module names and commands below are interfaces to implement, not claims about existing commands.

## 1. Migration rather than rewrite

Preserve the current `Symmetry_Cognitive_Maps` work as a pinned dependency or a versioned legacy directory with original license and provenance. Do not reorganize original files and change scientific behaviour in the same change set.

Create an `asset_inventory.csv` recording artifact role, original path, revision, SHA-256, model/environment configuration, usable status, and reason for any incompatibility. Include original checkpoints and trajectory files where accessible. A published endpoint checkpoint cannot supply earlier training states that were never saved; a weight-only checkpoint cannot support exact optimizer-state continuation.

Record the correspondence between the October 6 manuscript, public README, source revision, and actual checkpoint metadata. The README inspection supporting this plan was not a source audit. In particular, verify `parity`/`flip`, `s1`/`C1`, feature normalization, input noise, rendering, recurrent-state shape, and prediction-horizon semantics.

Build the legacy adapter first. A compatibility report should show one-step update, sequence update, predictive rollout, parameter-group assignment, gradient clipping, and optimizer-step equivalence on a small reference fixture. Numeric tolerances must state dtype and hardware. Deterministic audit mode can disable stochasticity, but the production stochastic path must receive a separate distributional or paired-RNG test.

## 2. Proposed repository structure

```text
map_dynamics/
  README.md
  pyproject.toml
  environment.lock
  CITATION.cff
  LICENSE
  AGENTS.md
  docs/
    RESEARCH_SPEC.md
    IMPLEMENTATION_PLAN.md
    SOURCES.md
    DATA_CONTRACTS.md
    decisions/
  configs/
    environments/
    models/
    learning/
    protocols/
    analyses/
  registries/
    experiments.yaml
    claims.yaml
    figures.yaml
    assets.csv
  src/map_dynamics/
    legacy/         # adapters and compatibility metadata
    envs/           # state, rendering, transitions, tasks, symmetry
    observers/      # exact and finite-window information benchmarks
    models/         # recurrent core and prediction/policy/value heads
    learning/       # predictive update, recurrent PPO, loss routing
    protocols/      # passive, forced route, active, schedule, yoke
    recording/      # immutable banks and complete state recordings
    analysis/       # decoding, components, dynamics, transfer, statistics
    interventions/  # subspace, state swap, cue and experience controls
    experiments/    # orchestration of registered X0–X5 protocols
    reporting/      # source tables, figures, claim audit
    cli.py
  tests/
    unit/
    scientific_invariants/
    regression/
    integration/
  scripts/
  artifacts/        # normally outside Git; paths through manifests
  results/          # derived tables and provenance, not handwritten data
```

Use one Python package, typed configuration, and a small number of explicit interfaces. Keep training and analysis separate. Notebooks may explore results but cannot be the only implementation of a manuscript figure. Do not add a web application, distributed service layer, or framework supporting unrelated environments in the first milestone.

Pin an actually tested software environment during implementation; do not invent a lockfile before dependencies have been installed and tested. Use PyTorch and the existing environment stack where compatibility permits, with NumPy/SciPy/scikit-learn for ordinary analysis and chunked arrays for recordings. These are proposed implementation choices, not claims of compatibility with any latest release.

## 3. Interfaces and ownership

### Environment and observation separation

`Environment.reset(seed, condition)` initializes a valid state and returns an agent-visible observation plus separate analyst metadata.

`Environment.step(action)` returns the next agent-visible observation, reward, termination/truncation flags, and analyst metadata.

`Environment.clone_state()` and `restore_state()` serialize all latent state and RNG needed for branching. `transform_state(g)` and `transform_action(g)` implement declared symmetries. `enumerate_states()` and sparse transition/observation interfaces support the exact observer for finite-state families.

`AgentObservation` contains only declared inputs. `AnalystMetadata` contains position, heading truth, context target, cue age, orbit ID, route information, and task labels. The policy call must not accept the latter type. Do not depend on convention alone to prevent leakage.

Observation timing and the inclusion of previous action/reward are versioned. Variable action masks are observations: either they are part of the permitted information channel and observer likelihood, or they remain invariant and are explicitly audited.

### Recurrent model

`RecurrentCore.initial_state(batch_shape, generator)` returns the complete recurrent state. `step(agent_observation, core_state)` returns the next complete state, population activity, and optional predictive outputs. Complete state may be more than one tensor; an LSTM control includes both hidden and cell state.

`PredictionHead` operates on a branch/copy of the live state. `PolicyHead` and `ValueHead` receive current population activity only in the primary model. `Intervention.apply` has an explicitly specified before/after-update position and records its action in an intervention log.

Use `StateLayout` metadata to distinguish all recurrent variables from the population activity selected for analysis. A state patch targeting one tensor must not accidentally leave another memory-bearing tensor unmeasured.

### Learning and protocols

The learner owns gradient updates, optimizer states, and counters. The protocol owns data collection and assignment of movement/task control. Forced-route navigation actions are exogenous; only genuinely agent-selected decisions enter the on-policy actor objective.

The loss router declares which objectives can update which parameters. Test gradients after backward, not only configuration flags. Stop-gradient prevents actor, critic, entropy, and policy-dependent regularizer gradients from entering the backbone while preserving predictive gradients.

Recurrent PPO needs episode boundaries, valid decision masks, old log probabilities, rewards, value targets, burn-in, and sequence masks. Treat reward scaling, advantage normalization, optimization epochs, clipping, and sequence length as explicit parameters. Cache identifiers alone are not sufficient when weights change: recurrent state replay must use the correct parameter version and documented approximation.

### Analysis

Every analysis accepts artifact manifests, fitting/validation/testing split IDs, resolved estimator parameters, and an output location. It returns a table and provenance record. Analysis functions do not load a private default checkpoint, change a training config, or mutate source recordings.

Every row carries its network, layout, condition, checkpoint, analysis version, and split provenance. A metric named `context_accuracy` without those keys is insufficient.

## 4. Data contracts

### Run manifest

Required fields: schema version; run ID; experiment ID; condition ID; pairing/branch ID; implementation status; source revision; dirty diff hash; dependency lock hash; source checkpoint hash; resolved configuration hash; environment layout hash; renderer version; training data/split hash; root seed; separately derived model/environment/action/noise/analysis seeds; hardware and dtype; deterministic mode; start/end status; and artifact checksums.

Configuration must distinguish optimizer updates, transitions, episodes, predictive target positions, actual informative encounters, and physical elapsed time. Partial jobs carry `incomplete` status and cannot masquerade as final checkpoints.

### Checkpoint

Store complete model parameters, all optimizer and scheduler states, RNG states, counters, training schedule position, input normalization, policy version, sequence/buffer position as needed, parent checkpoint, environment/config hashes, and schema version. Save atomically. An interrupted checkpoint must not replace the last valid one.

Test pause/resume against an uninterrupted small deterministic run. When exact continuation is unavailable, identify the missing state and label the resumed run accordingly.

### Trajectory bank

Index: bank ID, version, bank hash, split role, layout IDs, protocol, episode IDs, trajectory-family IDs, paired trajectory IDs, and raw versus generated representation.

Per transition: time index, allowed observation inputs after external noise, action, reward, terminal/truncated flags, state hash, analyst metadata, cue event ID, cue validity/value, evidence model version, and policy ID where endogenous. Record enough noise and renderer information to reproduce an input exactly. Do not quantize a weak cue through an undocumented uint8 path.

Store identifying metadata in a separate table or namespace, inaccessible to the policy data loader. Future observed actions used for prediction targets cannot appear in current action-selection input.

### Population recording

Index by run, checkpoint, diagnostic-bank hash, episode, and timestep. Store complete recurrent state where required and the selected population-activity tensor. Record before/after-intervention activity on intervention probes. Store policy probabilities and chosen action for functional comparisons.

Use float32 for quantitative reference recordings; any reduced-precision compression requires an error audit, especially for weak-context signals. Use chunked arrays with explicit dimensions, dtype, checksum, and valid-length masks. Prefer fixed diagnostic sampling over uncontrolled accumulation of every activation.

### Analysis result

Required fields include estimator name/version, train/validation/test banks, fitted preprocessing hash, network/layout IDs, target definition, common-support rules, sample counts, effective replicate unit, result, uncertainty method, flags, and source artifact hashes.

For cross-temporal analyses add train/test delay and train/test checkpoint. For transfer add source/target environment IDs, adaptation mode, independent initialization, task budget, criterion, and censoring status. For scaling add physical dimensions, cell size, speed, cue count/density, prediction horizon, discount, and memory-training window.

### Figure manifest

Each panel identifies analysis tables and columns, transformations, uncertainty calculation, display rules, and source-data output. No central figure value may be manually entered. Figure-generation code cannot silently exclude seeds or refit a model.

## 5. Mandatory test families

| Test family | Concrete failure it should detect |
|---|---|
| Input identity | A nominal symmetry changes a permitted observation or arrival signal |
| Transition equivariance | A rotated state/action pair has inconsistent next-state behaviour |
| Reward/termination leak | Correct-context information is available before the stated decision |
| Cue calibration | Nominal cue levels collapse under rendering, storage, or noise |
| Filter | Incorrect priors, transitions, normalization, reward likelihood, or action conditioning |
| Split isolation | Twin partners, trajectories, layouts, or temporal neighbors leak across evaluation splits |
| Decoder calibration | A path-ID code looks context informative, or a weak known signal is undetectable |
| Gradient routing | Detached task learning still changes the recurrent backbone through critic or shared loss |
| Prediction branching | Realized future movement changes the live online state or policy |
| RL timing | Current reward is used before the action, or reset/truncation bootstrapping is wrong |
| Recurrent replay | Minibatch sequence state differs from the intended forward history |
| Intervention | A zero dose changes states, dose has wrong timing, or test data chose the subspace |
| Resume | Restored optimizer/RNG state fails to reproduce a reference continuation |
| Ensemble | Batched training changes per-model gradients, clipping, or optimizer statistics |
| Provenance | A figure can be generated from missing/mismatched/handwritten source data |
| Registry | A hypothesis is marked supported without results and required controls |

Synthetic population fixtures should include: a constant code; a path-ID-only code; a perfectly shared spatial code; a shared spatial code with weak persistent context; a rotating context direction; context decaying in noise; and a context-rich code connected to a deliberately context-blind policy. The analysis should distinguish these cases before it is interpreted on learned activity.

A recurrent representation with encoded context but a context-blind policy should pass population decoding and fail the behavioural readout test. This fixture is particularly important: it prevents the repository from automatically identifying decodability with use.

## 6. Issue-sized implementation stages

### M0 — Audit and adapter

Deliver asset inventory, source-version compatibility matrix, exact legacy adapter, checkpoint loader, and regression fixtures. Run no new sweep. Exit when a selected original checkpoint gives traceable observations, hidden trajectories, and reference metrics.

### M1 — Information and diagnostic infrastructure

Deliver symmetric/noisy-cue toy environments, validated sparse observer, permitted-input types, split-safe banks, population recorder, and synthetic-code tests. Exit when the observer benchmarks and analysis nulls pass and a single end-to-end report can be regenerated from its manifest.

### M2 — Forced-route learning

Deliver transient cue/delay/decision task, predictive/task/joint loss routing, recurrent learner, and a short resume test. Run the registered eighteen-network development pilot only after correctness tests. Exit when the task provides learning signal, positive controls function, and observed failure modes can be separated—not only when reward rises.

### M3 — Population dynamics

Deliver cross-position/time/checkpoint decoding, factorial components, full-update Jacobian response checks, targeted and control perturbations, and reduced-model comparison. Exit when synthetic stable/rotating/decaying/use-failure codes produce distinct expected diagnostics and learned-model estimates have held-out support.

### M4 — Active evidence-seeking

Deliver optional checking, active/stop-gradient modes, policy trajectory logging, branchable intervention states, and controlled-evidence schedules. Exit when checking has a quantified reward trade-off, policy evidence access changes as measured, and valid protocols separate direct gradients from experience effects.

### M5 — Transfer, schedules, scale

Deliver checkpoint branching, frozen versus adaptive transfer, censored learning curves, pre-transfer predictors, schedule hashes, and separate physical/context/resolution scale configs. Exit when a frozen predictor is evaluated on untouched layouts and the simpler baselines are reported alongside it.

### M6 — Release and paper audit

Deliver a clean-install reproduction path, source tables, minimal recordings/checkpoints, archived code/configs, claim-by-claim report, and documented negative results. Distinguish successful software execution from scientific support.

## 7. Proposed command surface

The following are target interfaces, not commands implemented by this design package:

```text
map-dynamics audit-legacy --manifest assets.csv
map-dynamics validate --experiment X1 --config configs/protocols/forced_delay.yaml
map-dynamics make-bank --config configs/protocols/diagnostics.yaml
map-dynamics run --experiment X1 --manifest frozen_campaign.json
map-dynamics record --run RUN_ID --bank BANK_ID
map-dynamics analyze --analysis temporal_decode --manifest ANALYSIS_MANIFEST
map-dynamics audit-claims --registry registries/claims.yaml
map-dynamics replot --figure F3 --manifest RELEASE_MANIFEST
```

Use separate command modes for smoke tests, pilot, and frozen confirmatory campaigns. A submission command checks dependencies, hashes, gate status, and compute budget before starting. It must never launch a full Cartesian sweep merely because a config contains several arrays.

## 8. Definition of the first useful repository

The initial repository is useful when a new contributor can install a tested environment, load one original model, generate or load one paired diagnostic bank, verify its information structure, record the population, distinguish context decoding from task use, and reproduce one source-linked report. A large directory tree with mocked training functions does not meet that definition.
