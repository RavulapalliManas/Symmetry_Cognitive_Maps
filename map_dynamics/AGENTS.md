# Rules for implementation contributors and coding agents

This file governs development of the proposed computational study. It does not assert that the study has been implemented.

1. Read docs/RESEARCH_SPEC.md, docs/IMPLEMENTATION_PLAN.md, and docs/SOURCES.md before editing scientific behaviour. Preserve the computational-only scope; do not make animal-data acquisition a dependency.
2. Preserve a pinned legacy implementation. Never silently replace the pRNNth cell, normalize a signal differently, reinterpret a checkpoint, or equate `parity` with `flip` without a tested mapping.
3. Implement the first complete vertical slice before adding architecture families, dashboards, or large sweeps. Never substitute plausible mock metrics for model outputs.
4. Keep agent-visible inputs structurally separate from analyst metadata. Account for reward, termination, action masks, initial state, previous action, and future-target leakage.
5. A chance decoder is not proof of no information. Run exact controls, path-ID nulls, weak-signal calibration, and fixed/refitted readouts. Fit all preprocessing on fitting data only.
6. Recurrent memory includes every state variable, not only the tensor selected for plots. Prediction rollouts must not mutate the live state.
7. Do not apply an uncorrected on-policy policy-gradient loss to a yoked off-policy trajectory stream. Declare the objective and information available to every comparator.
8. Every experiment needs an ID, frozen config, dependencies, estimand, controls, split manifest, replicate unit, and failure interpretation. A new analysis after viewing final results is exploratory unless evaluated on new held-out data.
9. Report failed learning runs, nonconvergence, negative results, and missing artifacts. Do not expand seed counts until a desired p-value appears, tune thresholds to reproduce manuscript values, or select units/trajectories for visual appeal.
10. Save complete optimizer/RNG/protocol state for continuation. Record source/config/data hashes. Never state that a test passed unless it was executed and its output is available.
11. Run scientific invariants before training. Profile a small job and obtain campaign authorization before launching expensive sweeps. Do not infer available hardware or current budgets from older projects.
12. Separate `replot`, `reanalyze`, and `retrain`. Every manuscript number must trace to source artifacts and a deterministic analysis recipe.
13. Keep hypothesis status separate from implementation status. Software tests establish implementation properties, not support for biological or computational hypotheses.
14. Claims of attractors, irreversible development, critical periods, or general scaling laws require their specific evidence. Do not rename ordinary clustering, slow adaptation, or a fitted curve as a stronger phenomenon.
15. Document substantive design changes in docs/decisions with rationale, affected hypotheses, migration implications, and required regression tests.

## Checkout paths and preparation boundary

- This package lives in `map_dynamics/`. The parent repository owns the legacy source.
- The supplied design files remain unchanged under `docs/` and `registries/`.
- `README.md` owns current setup status and unresolved requirements.
- `registries/experiments.yaml` and `registries/claims.yaml` retain the proposed scientific statuses.
- `configs/protocols/X0.yaml` through `X5.yaml` contain preparation templates, not executable scientific protocols.
- Use `uv sync --locked --extra science` on cpu-angad.
- Run `MAP_DYNAMICS_LEGACY_ROOT="$PWD/legacy_reference" uv run --locked --extra science pytest -q tests`.
- Use `uv run --locked map-dynamics --root . validate` to check preparation inputs.
- The CLI prepares drafts, inventories source files, and runs `replay-legacy`. `develop-x1` runs bounded CPU engineering checks; it cannot launch a scientific campaign or mark hypotheses supported.
- `configs/protocols/X1_development.yaml` is a development protocol. Its caps must not be expanded into a study budget.
- Run `uv run --locked --extra science pytest -q tests/test_x1.py` for the forced-route task and learner.
- Preserve `legacy_reference/` byte-for-byte; its manifest pins the tested source.
- `configs/protocols/X0_replay.yaml` is an executable reference config; X0–X5 remain preparation templates.
- Do not add empty modules for later milestones or mock experimental outputs.
- Store recordings, checkpoint files, and generated plans on the fleet, outside this checkout.
