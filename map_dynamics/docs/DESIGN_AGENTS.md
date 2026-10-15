# Rules for implementation contributors and coding agents

This file governs development of the proposed computational study. It does not assert that the study has been implemented.

1. Read RESEARCH_SPEC.md, IMPLEMENTATION_PLAN.md, and SOURCES.md before editing scientific behaviour. Preserve the computational-only scope; do not make animal-data acquisition a dependency.
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
