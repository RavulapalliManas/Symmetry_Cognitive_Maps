# Learning when to share a cognitive map
## Research and repository specification

**Version:** 0.1 — 15 October 2026  
**Scope:** Computational neuroscience only.  
**Target:** Nature Neuroscience; this is a research ambition, not a prediction of acceptance.  
**Status:** Proposed research design. No new simulations, replication tests, causal experiments, or repository migrations have been run for this specification.

## 1. Project definition

This project will investigate how recurrent neural populations learn to preserve reusable spatial structure while maintaining the contextual distinctions needed for behaviour. Agents will learn in environments whose sensorimotor symmetries, informative cues, geometry, task requirements, and opportunities to reacquire information can be manipulated independently. We will measure how population coding develops during training, how context information evolves within an episode, what downstream policies actually use, and how the resulting behaviour changes subsequent experience and learning.

The central question is:

> Under what conditions does an agent learn to share a map, differentiate it, or compensate behaviourally for its ambiguity—and can those outcomes be predicted from evidence access and memory dynamics?

The central explanatory proposal is not that symmetric environments necessarily impair learning. Sharing a representation can facilitate compatible tasks, while interfering with tasks requiring context-specific distinctions. Context may be encoded as an additional reusable component, through position-specific interactions, through more extensive remapping, or through a changing temporal code. Behavioural evidence-seeking can supplement any of these organizations.

The project will therefore distinguish three levels: what the agent could infer from its available history; what information is recoverable from its recurrent population; and what information its current policy uses. A fourth level—how behaviour changes future access to evidence—closes the learning loop.

The intended contribution is a tested computational mechanism connecting population development to transfer and information-seeking. It is not a navigation leaderboard, an atlas of embeddings, or a claim that a generic artificial recurrent unit is literally a hippocampal neuron.

## 2. Evidence, inheritance, and boundaries

### 2.1 What the manuscript supplies

The attached manuscript reports 18 x 18 MiniGrid arenas with C1, C2, and C4 landmark symmetries; a layer-normalized pRNNth with 500 hidden units; 200-step truncation; and a five-step prediction horizon. Views have 147 components and the movement input has five. The reference training uses RMSprop with distinct parameter-group learning rates, clipping, and other details that must be inherited rather than approximated [M, pp. 14–16].

It reports contrasting outcomes for matched one-bit heading channels called `axis` and `flip`, an exact no-evidence symmetry control, and a separate repeated-room setup in which informative corridor experience can make context knowable. In translated rooms, the manuscript reports a roughly fifteen-step context-retention half-life, a targeted room-difference-subspace perturbation, and a room-report positive control. Figure 5 on page 9 is the clearest existing empirical foundation for the proposed study of available-but-lost information [M, pp. 5–9, 19–20].

Its frozen-code readout experiments establish task-dependent consequences of sharing, but do not establish co-development of a policy and its representation [M, pp. 8–9, 17–18]. Exact equality of states driven by identical histories is also present without learning; it must not be presented as a developmental finding [M, p. 16].

### 2.2 What the public repository supplies—and what remains unaudited

The manuscript's `Thesis_work` link currently redirects to `RavulapalliManas/Symmetry_Cognitive_Maps`. The public README describes `project5_symmetry/environments`, `experiments`, `analysis`, and `evaluation`, with the recurrent implementation under `utils/Architectures.py` and `utils/thetaRNN.py`. It also lists ensemble training, tests, committed result tables, and some existing size controls [R]. These are README-level observations, not a completed source-code or execution audit.

The README uses `parity` where the manuscript uses `flip`, and `s1/s2/s4` in several commands where the paper uses `C1/C2/C4`. Do not silently equate names. Verify the encoded matrices and environment semantics, then store a tested alias map. Its declared size controls should be inventoried before calling a new scale sweep novel.

Preserve a pinned legacy revision and its result artifacts. Build new interfaces around the legacy model before refactoring it. The submitted PDF, public README, code revision, and checkpoint metadata may correspond to different snapshots. An explicit compatibility table must record those differences rather than silently harmonize them.

### 2.3 New hypotheses are not inherited results

The following remain hypotheses: task relevance stabilizes context coding; context differentiation spreads from cues or decisions; early shared structure changes later learnability; cheaper checking reduces the need for durable memory; and independent retention measurements predict behaviour in larger or rearranged environments. Every hypothesis requires an experiment that could reject it.

## 3. Scientific aims and claim gates

### Aim 1 — Identify the organization and development of context-sensitive population coding

Determine whether a shared spatial component is preserved while context coding is added, localized, recoded, or more broadly reorganized. Track the same network through training using fixed diagnostic histories. The primary evidence is held-out decoding, cross-condition generalization, and targeted functional perturbation—not visual separation in an embedding.

### Aim 2 — Identify dynamics responsible for maintenance, loss, and use

Discriminate contraction, temporal recoding, interference, and downstream use failure. Measure context trajectories after matched cues, temporal decoder transfer, local recurrent responses, perturbation recovery, and decision performance. A larger hidden-state distance is not itself better memory, and a chance decoder does not prove that no information exists.

### Aim 3 — Test the feedback between evidence-seeking and population development

Manipulate the cost or availability of informative encounters. Determine whether the agent maintains context, refreshes it behaviourally, or fails to acquire either solution. Use randomized interventions and controlled experience to test whether altered later evidence contributes to lasting learning effects.

### Aim 4 — Predict generalization and history-dependent learnability

Ask whether population measurements taken before a new task predict zero-shot transfer, frozen-readout learning, adaptation cost, and interference. Test independent changes in physical extent, repeated-context count, and discretization. A held-out-environment prediction is stronger than a separate fit in every arena.

Each claim must declare its source runs, estimator, independent test split, uncertainty, decisive controls, failure outcomes, and current status. Passing a software test never changes a scientific hypothesis to “supported.”

## 4. Formal system and information contract

### 4.1 Environment

Represent the latent world as s_t = (position, heading, room/context, task phase, cue state, any remaining transition-relevant variables). Context can be derived from physical state; do not add an independent context variable unless the generative model actually has one. Different environment families may use different valid state factorizations.

At step t the agent receives its current rendered observation and declared arrival signals. It updates its recurrent state, selects an egocentric action, and receives the next observation, realized movement signals, reward, and termination information. Fix the indexing convention in code and documentation before implementing loss functions.

For the new active agent, use a declared input vector containing current view, realized speed, encoded heading, previous egocentric action, and previous reward. This is an explicitly versioned extension of the legacy interface, not an exact reproduction of its input dimensionality. All learning-mode comparisons use the same new input vector. Any additional task instruction must be defined in the task contract and must not disclose the correct hidden context.

The primary policy and critic use the same recurrent state. Initially they have no additional recurrent memory and no privileged state input. Analyst metadata, Bayes beliefs, true coordinates, cue age, context labels, and shortest-path distances are excluded from their observation dictionaries. A privileged-state control is a separately labelled agent.

### 4.2 Timing contract

For each transition, record `obs_t`, `motion_t`, `prev_action`, `prev_reward`, `state_before_update`, `state_after_update`, `action_t`, `reward_t`, `terminated_t`, `truncated_t`, and identifiers. `reward_t` is received only after `action_t` and cannot appear in the state used to select that action.

The observer and diagnostic system see exactly the external history permitted to the agent. When the model omits a signal, label a bound using additional signals as privileged. Distinguish actual noisy observations from an ideal clean-sensor upper bound.

### 4.3 Exact observer

Implement a sparse finite-state Bayesian filter for each environment family. Condition on executed actions; do not multiply by a state-dependent policy likelihood when the action was chosen from the same observed history. Include stochastic transitions, noisy cues, observed rewards, episode boundaries, visible action masks, and any other permitted evidence in the likelihood.

For a context target C_t, report the mean Bayes classification benchmark:

A_Bayes = E[max_c P(C_t=c | available history)].

Validate on an exactly symmetric no-evidence world, a deterministic unique-cue world, and a noisy binary-cue world with known posterior. Probability mass must sum to one, impossible transitions must receive zero probability, and unrecognized observation models must fail rather than silently fall back to a different filter.

Correct feedback can identify context after a decision. Thus the no-cue impossibility claim applies to the audited pre-feedback history, not automatically to subsequent trials. Randomize initial context and reset recurrent memory independently of context when that is the intended protocol.

### 4.4 Symmetry audit

A symmetry certificate must test transition equivariance and equality of all declared observation channels under paired trajectories. Geometry alone is insufficient. Audit start distributions, headings, reward before commitment, episode duration, termination, action masks, collision signals, textures, random seeds, and identifiers accessible to the learner.

A learned policy may choose an arbitrary orientation and still fail to identify the correct absolute context. Broken symmetry in behaviour is not evidence that the observations contained the correct label.

## 5. Environment families and scaling

### E-A: Legacy reference

Reproduce selected C1/C2/C4 and heading-channel controls with the exact legacy interface. Reuse existing valid checkpoints and trajectories where available. This is the regression anchor, not the new paper's entire main result.

### E-B: Transient-cue arena

Extend the 18 x 18 C2 setting with a brief identifying cue at symmetry-related sites. Apart from that cue and task feedback at commitment, matched trajectories must remain sensorimotor-equivalent in the intended delay period. Use an ambiguous heading condition for the principal memory assay; continuously informative absolute heading would undermine the transient-memory manipulation.

For forced-delay probes, restrict paths to corresponding domains or explicitly propagate changing orbit-member labels. Never assume that the current half-of-arena label stays fixed when the path crosses a domain boundary. For free movement, use the full transition model rather than a persistent-bit shortcut.

The first graded cue should be a calibrated binary symbol with reliability q, implemented without color-quantization ambiguity. Use q = 0.95 as a proposed primary pilot level, with q = 0.75 as a later stress condition and q = 0.5 as an uninformative control. These are design defaults, not optimized or confirmed parameters. Independent cue noise must be balanced across contexts.

### E-C: Repeated rooms

Retain translated and rotated-room variants as a complementary family. The existing corridor creates evidence and its geometry differs between arrangements; do not use it as though it were an exact global symmetry. New route comparisons should match route length when route length is not the manipulated factor. Store actual informative junction encounters, not just room entry times.

### E-D: Active checking

Provide symmetry-related informative beacons accessible through optional routes. Entering a beacon region yields the specified cue; the beacon's visible location and trigger rules are symmetric while its cue value carries context. The primary manipulation changes detour length under a common movement cost. A secondary condition may change an explicit sensing cost while holding paths fixed; do not pool these manipulations.

There should be no bonus for cue visits in the primary task. Checking must help through improved task reward. Use equally long noninformative detours and matched cue availability as controls.

### Three distinct scale axes

**Physical extent:** 18 x 18 to 36 x 36, with 72 x 72 conditional on profiling and intermediate results. Keep movement step and view range fixed; contrast fixed informative-cue count with fixed cue density.

**Repeated-context count:** two to four to eight rooms with local room size fixed. This tests contextual expansion and reuse, not only longer trajectories.

**Resolution:** change cell resolution while holding physical arena extent, physical velocity, sensor extent, cue dimensions, and time units fixed. This is a discretization control. Sensor resampling must be validated so it does not create identifying artifacts.

Report results under both equal interaction budgets and selected equal-evidence/coverage controls. Account for prediction horizon, discount horizon, episode timeout, and backpropagation length. With changed physical timestep, a continuous-time discount convention such as gamma = exp(-kappa * dt) can preserve a time-based horizon, but its kappa is a declared modelling choice.

## 6. Models and learning regimes

### Reference model

Use an adapter around the legacy 500-unit pRNNth. Preserve its real update, noise, normalization, rollout semantics, initialization, and optimizer behaviour in the reproduction experiment. Do not replace it with a generic tanh RNN and retain the same name.

### Main active model

Use the same recurrent-cell family with the versioned active input contract, a predictive output head, a feedforward policy head, and a feedforward value head. Recurrent PPO is the proposed initial task-learning implementation [PPO]. Its role is to provide a controlled task-learning signal, not to constitute the scientific contribution. Select numerical hyperparameters in pilots on development layouts and then freeze them.

Use one explicitly defined optimizer/update schedule for the combined new model. Changing the optimizer from the legacy reproduction is allowed only as a documented new-protocol choice. Do not silently double-update shared parameters with independent prediction and policy optimizers. Log prediction, actor, critic, entropy, and regularization contributions separately.

Loss routing must support:

| Mode | Population receives | Policy learning | Interpretation |
|---|---|---|---|
| Predictive-only | Prediction gradient | Frozen diagnostic readout fitted separately | What prediction installs |
| Joint | Prediction, actor and critic gradients | On-policy | Joint representation and task learning |
| Task-only | Actor and critic gradients | On-policy | Is prediction necessary for the proposed mechanism? |
| Predictive + stop-gradient actor | Prediction gradient only | Actor/critic read detached population state | Indirect effect of experience without direct task gradients into the population |
| Frozen population | No recurrent updates | Actor/critic only | Readout adaptation without representational learning |

The full mode set need not run in every experiment. Primary forced-route pilots compare predictive-only, joint, and task-only. The stop-gradient mode is essential in the active-feedback experiment. Room-report supervision remains a labelled positive control for representational capacity, not the principal explanatory model.

On-policy rollouts must retain old action probabilities, current policy version, sequence boundaries, and all states needed for correct recurrent replay. Recompute recurrent states with documented burn-in under updated weights; measure sensitivity to burn-in length. A truncated rollout is not an environmental terminal. Multi-step prediction may use realized future movement from an already collected trajectory as a conditional training target, but those future signals must never enter the online policy or live hidden state. Use a separate prediction branch.

### Mechanistically informative controls

Start with a memoryless observation model, a frozen/random recurrent population with trainable heads, a finite-window observer, the full-history Bayesian benchmark, and a GRU replication of key learned effects. Add a deliberately factorized shared-space-plus-context architecture only as a constructive reference; its imposed factorization cannot demonstrate that an unconstrained network discovered one.

Low et al. provide a prior example of aligned remapping in navigation/context-inference RNNs [L]. Cone and Clopath provide a prior example of representation–behaviour co-development with plateau-based plasticity [C]. Implement their original canonical task before claiming an implementation is a faithful reproduction; adaptations to this environment must be labelled adaptations.

A spiking model, multiple anatomically named regions, and exhaustive architecture comparisons are not required in version 0.1. Any later biological detail must distinguish a mechanism rather than decorate the model.

## 7. Longitudinal recording and population analyses

### 7.1 Three clocks

Record training age in environment transitions and parameter updates, within-episode time in steps and physical units, and time since the last informative encounter. Do not substitute episode count for all three. Frozen diagnostic recording occurs at fixed checkpoints, including initialization, early learning, established performance, and continued training.

Use a dense early/log-spaced checkpoint schedule. A proposed pilot is 0, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, and 64000 optimizer updates, extended if necessary. This schedule is not a required training budget. Checkpoints must also record actual environmental interactions and cue encounters so that different learning modes can be aligned meaningfully.

### 7.2 Diagnostic banks

Maintain separate immutable banks for matched twin histories, cue-delay probes, standardized navigation, perturbation probes, and transfer tasks. Full trajectories and all symmetry-related partners belong to one split. Fit standardization, dimensionality reduction, decoder regularization, and intervention subspaces on fitting splits only. Selection/tuning layouts and final evaluation layouts must be distinct.

The policy-generated trajectory bank is separate. It is used to measure natural strategy, not to estimate a population change without conditioning on changed experience. Every evaluation bank has a hash and version; changing it invalidates previous comparisons unless explicitly reconciled.

### 7.3 Shared and context-sensitive components

For balanced matched states, define h+ = (mean(h_A)+mean(h_B))/2 and h- = (mean(h_A)-mean(h_B))/2. This is an analysis decomposition, not proof of two neural modules. Use independent trajectory halves for cross-validated signal-energy estimates and report uncertainty rather than truncating negative finite-sample estimates to zero.

Fit an identifiable factorial encoding model of local position/heading, context, cue age, and their interactions. Use sum-to-zero or equivalent constraints, balanced support, and held-out predictive variance. Lack of common support is a failure of the comparison, not a reason to invent matched data.

Primary questions are whether context coding transfers across positions and routes, whether spatial readouts transfer across contexts, and whether context-dependent interactions concentrate near relevant decisions. Cross-condition generalization follows the operational logic of Bernardi et al. [B]; ordinary in-distribution decoding alone does not establish abstraction.

### 7.4 Available, recoverable, used

For each checkpoint and delay report Bayes benchmark, memoryless-input control, linear and calibrated nonlinear population decoding, fixed-policy choice accuracy, and a freshly fitted diagnostic readout. Decode true context and, in noisy-cue conditions, distinguish it from merely decoding the observed cue symbol.

A normalized accessibility index can be reported as (A_population - A_memoryless)/(A_Bayes - A_memoryless) only when the denominator exceeds a declared minimum. Keep the unnormalized accuracies primary, do not call this mutual information, and do not clip noisy estimates into [0,1].

A decoder is an accessibility lower bound. Rejecting “no difference” does not establish a useful amount of information; failing to reject it does not prove absence. Include synthetic weak-signal calibration and equivalence bounds appropriate to the behavioural effect size.

### 7.5 Temporal and developmental decoder transfer

Construct a matrix by training a context decoder at one delay and testing at other delays. Repeat across training checkpoints. A stable diagonal-only pattern can be consistent with temporal recoding, while broad transfer supports a reusable readout. Neither conclusion follows from a single refitted decoder.

Fixed decoders retain their original centering and scaling; otherwise the fixed-readout condition has secretly adapted. A cross-checkpoint alignment analysis is separate and must not be described as fixed biological readout stability.

### 7.6 Spatial development of differentiation

Map held-out context discriminability at each position over training. Compare cue distance, decision distance, experienced transition distance, elapsed delay, and local exposure as predictors. Move cues and decisions independently and introduce alternative routes. Change speed to separate elapsed time from distance.

Onset should be a sustained, uncertainty-qualified increase above a calibrated floor, with censoring when no onset occurs—not the first significant p-value in hundreds of bins. Thresholds and smoothing are frozen using pilot data. A spatial progression of coding need not imply literal propagation of plasticity; local exposure and credit assignment are competing explanations.

### 7.7 Population size and recording size

Distinguish training networks of 250, 500, and 1000 units from recording subsets of one trained network. Use the former to study resource-dependent learning and the latter to study sampling of the code. Fit PCA and effective dimensionality on independent data with matched sample size; dimensionality is not in itself a measure of computational quality.

For single-unit fields, use occupancy and heading controls, all-unit summaries, and example-cell selection rules fixed before inspecting effect strength. Artificial hidden activations are not spikes or calcium signals. Synthetic observation noise or subsampling is a recording-model sensitivity analysis, not a biological dataset.

## 8. Dynamical mechanism tests

### 8.1 Matched-input perturbation dynamics

After different context cues, supply matched subsequent inputs. For small differences, test the local approximation delta_h[t+1] = J_t delta_h[t], where J_t is the Jacobian of the complete recurrent update. Include normalization, gating, and all recurrent state components. Compare predicted and observed finite perturbation responses over a declared local radius.

Estimate amplification/contraction, rotation of context-relevant directions, and their overlap with policy-sensitive directions. Under stochastic dynamics, compare common-noise and independent-noise probes: the first isolates conditional dynamical sensitivity, the second measures functional reliability. Neither alone is a complete memory explanation.

Do not infer an attractor from an embedding. Attractor-like claims require appropriate stationary inputs, bounded perturbations, recovery tests, and comparison with simple slow-decay alternatives. Within-episode convergence is distinct from optimization convergence across training.

### 8.2 Competing reduced explanations

Fit and compare at least a cue-age decay model, a cue-age-plus-interference model, and a population-dynamics-informed model. A useful first reduced model is m[t+1] = rho*m[t] + beta*e[t+1] + xi[t], where evidence, persistence and noise have separate roles. These parameters are fitted on designated calibration data, not on each test geometry.

A local Gaussian approximation may propagate mean differences and covariance separately:

Delta_mu[t+1] = J_t Delta_mu[t]

Sigma[t+1] = J_t Sigma[t] J_t^T + Q_t.

With balanced Gaussian classes and equal covariance, d-prime squared equals Delta_mu^T Sigma^-1 Delta_mu, and optimal linear accuracy equals Phi(d-prime/2). Test those assumptions rather than treating the formula as universally valid. A decreasing state norm without reduced signal-to-noise is not necessarily forgetting.

### 8.3 Selective intervention

Estimate a context-sensitive subspace on fitting trajectories. Test dose-dependent suppression or context swapping on held-out matched histories. Include no-op, rank-matched random, removed-energy-matched, spatial-subspace, and cue-removal controls. Use local position-conditioned centering where justified, and report perturbation magnitude and distance from naturally encountered states.

Apply interventions once to test maintenance/recovery, and throughout a delay to test ongoing dependence. These answer different questions. After perturbation, assess both an unchanged decoder and a refitted decoder. Loss of the original decoder can reflect recoding rather than removal of all context information.

An intervention can be targeted by analyst metadata; this does not authorize sending that metadata to the ordinary policy. Claims of context-specific causal dependence require preservation of relevant spatial, sensory-prediction, and motor controls, with off-manifold disruption treated as a possible explanation.

## 9. Experiment program

### X0 — Legacy audit and exact controls

Inventory source revision, environment assets, checkpoints, trajectory hashes, and report tables. Load selected legacy checkpoints through the adapter. Verify inputs, one-step updates, multi-step rollout outputs, losses, and—where applicable—per-model gradients and optimizer updates. Use declared numerical tolerances on the tested hardware; do not promise cross-platform bit identity.

Acceptance: reference effects remain compatible with their original estimator and metadata, while the new leakage-resistant pipeline is separately validated. A discrepancy is documented and investigated, never fixed by adjusting a threshold to match the paper.

### X1 — Context retention and its development under controlled experience

Use transient cues followed by externally controlled, matched delay trajectories and a final two-choice decision. Train over a common delay distribution; evaluate at 4, 8, 16, 32, and 64 steps plus held-out route variants. Use q = 0.95 in the first pilot. Compare shared and context-specific rules, and predictive-only, joint, and task-only learning.

Joint and task-only agents choose the final action and learn from its reward; imposed movement actions do not receive an on-policy actor loss. The predictive-only model receives no task-learning gradient and is evaluated by separately fitted readouts. A context-supervised positive control establishes that the architecture can carry the required distinction.

Primary outcomes: held-out decision accuracy against delay; population accessibility versus Bayes benchmark; cross-position and temporal decoder transfer; spatially resolved development of coding. Paired initializations and common experience banks support controlled contrasts. All agents see identical final-feedback conventions.

A first pilot uses three seeds per condition on one development layout: six training conditions, eighteen networks. This is a feasibility screen, not the confirmatory sample size. The graded-q stress test is gated on a reliable main assay.

### X2 — Dynamics, perturbations, and routes

Use X1 checkpoints. Test matched-delay routes with different observation interference; cue/decision relocations; one-time and sustained context perturbation; fixed-versus-refitted readouts; and reduced-model prediction. Most measurements require evaluation, not retraining every configuration.

Primary comparison: which proposed mechanism predicts held-out delay curves and perturbation effects? A mechanism that fits intact states but misses its declared perturbation prediction fails the stronger claim.

### X3 — Active remembering versus checking

Allow free navigation and optional cue reacquisition. First use the context-specific task with low/high checking costs and joint, stop-gradient, and task-only models. Add a shared-rule control to test whether checking is context-demand-specific. Keep cue reliability, reward timing, and the ordinary sensory interface fixed.

Measure task reward, accuracy, path cost, beacon visits, cue-age distribution at decisions, standardized-probe retention, and population organization over training. Better natural performance with unchanged standardized retention is consistent with compensation, not evidence for a stronger internal memory.

Yoked trajectory comparisons primarily identify the effect of the resulting data distribution on predictive representation learning. Do not train a replayed behaviour stream with an uncorrected on-policy PPO objective and interpret the comparison causally. Use valid behaviour-probability corrections or an explicitly specified common offline objective for any policy-learning yoke; report it as a different protocol.

The feedback experiment branches identical checkpoint/optimizer states into randomized context-perturbation and sham arms. After the acute perturbation ends, compare unrestricted future evidence with a controlled schedule supplying matched informative encounters and matched-duration noninformative events. No-perturbation arms receive the same schedule intervention. Test the perturbation-by-evidence-control interaction. This tests contribution of altered evidence exposure, not an automatically identified natural mediation effect.

### X4 — Experience order and later learnability

Construct early, late, and interleaved schedules using the same multiset of valid experience blocks. Use a fixed common evaluation/adaptation phase. First isolate schedule under externally specified histories; test endogenous exploration later, since identical experience counts alone do not match order or route content.

Compare optimizer-state continuation and reset controls, and ensure equal effective opportunity on the final task. Match task availability explicitly: task phases may use separate known readout contexts so that context requirements are not secretly announced by changing a hidden reward rule with no instruction.

Measure later interactions-to-criterion, old-task interference, and population reorganization. Report learning history dependence unless stronger evidence justifies a critical-period or hysteresis claim. This aim is gated: it is not necessary to force a developmental story if X1–X3 identify a different central mechanism.

### X5 — Predictive tests of transfer and scale

Before transfer, measure code descriptors and freeze the prediction model. Branch checkpoints into frozen-representation readout learning, full adaptation, and learning-from-scratch controls. Test compatible shared rules, context-specific exceptions at new locations, new appearance, altered connectivity, new rooms, and larger extent as distinct tests—not a single aggregate OOD score.

Use a small predeclared predictor with nested validation on development layouts and an untouched layout-family test set. Predict adaptation cost and interference, not only endpoint reward. Nonconverged runs are censored or explicitly counted as failures; they cannot disappear from mean learning curves.

A proposed first behavioural prediction is A_hat = mean_i a(Delta_i, u_i), combining a retention/choice curve calibrated on standardized trajectories with cue ages and declared route features from active decisions. Its assumptions are conditional transferability and adequate calibration support. Compare it against size-only, path-length-only, and cue-age-only models. Record where route-dependent interference or strategy changes invalidate it.

For one isolated decision with correct-choice reward R and a checking cost c, checking is useful when R*(A_check - A_memory) > c. Treat this as a benchmark with explicit assumptions. Sequential navigation requires a value-of-information comparison that includes travel time and future consequences, not this one-step inequality alone.

## 10. Statistical and computational design

The trained network and independently generated layout are replication units for their corresponding claims. Timesteps, units, trajectories, and repeated checkpoint measurements are nested observations. Use paired comparisons for cloned initializations or checkpoint branches, clustered/hierarchical uncertainty for layouts and networks, and trajectory-level resampling within a network where needed. Pairing must be respected in bootstrap and permutation procedures.

Pilot seeds choose valid operating ranges and estimate variance. Confirmatory layout/seed counts and a smallest meaningful effect are frozen afterwards; they are not repeatedly extended until significance. A provisional planning template is four development/confirmation layouts with eight paired seeds where affordable, but it is not a power calculation and the final split must leave genuinely unseen layouts for predictive tests.

Define a limited set of primary contrasts and control multiplicity within families. Report effect sizes and uncertainty alongside p-values. Predefine exclusions for numerical or infrastructure failure; retain scientifically unsuccessful agents. Track time-to-criterion with nonconvergence handled explicitly. Audit decoder power using synthetic codes at weak as well as strong effect sizes.

Profile one representative job before estimating the campaign. Record wall time, transitions/s, optimizer updates/s, peak memory, recording throughput, and run failure rate. GPU time cannot be inferred reliably from a model's parameter count or a previous different experiment. Batch independent networks only after verifying per-network optimizer state, gradient clipping, loss reduction, and noise semantics.

Record full population activity at fixed diagnostic checkpoints rather than every training step indefinitely. For illustration, 24 checkpoints x 50,000 samples x 500 units x 4 bytes is 2.4 GB of raw float32 activity per run, before metadata or compression. Most steps should store compact scalars; deep dynamics recordings and Jacobians should be restricted to selected independently chosen probes. This is a storage calculation, not measured performance.

## 11. Computational neuroscience interpretation

No new animal experiment or mandatory neural-data reanalysis is part of the scope. Maintain a literature constraint table stating the published phenomenon, its measurement level, the relevant model observable, and where the model fails. Separate motivation, qualitative consistency, quantitative reproduction, and genuinely new predictions.

Closest precedents prevent overclaiming: aligned context maps in an RNN are not novel by themselves [L]; joint behavioural and representational learning is not novel by itself [C]; and cross-condition decoding has an established operational interpretation [B]. The distinctive target is independent control of evidence and action, mechanistic discrimination of memory versus access versus use, and prospective prediction of transfer and compensation.

Potential prospective predictions include a cost-dependent trade-off between checking and persistent context coding; different route-transfer consequences of stationary and sequential context codes; and selective changes in context use despite preserved local spatial coding. These remain conditional model predictions, not observations about biological hippocampus.

Nature Neuroscience explicitly includes computational modelling within its scope while prioritizing fundamental insight into the nervous system [N1]. Repository quality supports scrutiny but does not substitute for that insight or establish likely acceptance.

## 12. Paper and release structure

A coherent manuscript could have six principal figure roles: (1) information/task dissociation and audited environment; (2) longitudinal population organization; (3) dynamical mechanism and causal perturbations; (4) active checking and the learning feedback; (5) prospective transfer/history prediction; (6) held-out scale and alternative-mechanism tests. These are question-to-evidence slots, not predetermined positive figures.

Retain a negative-result route. The strongest outcome may be a failure of the simplest factorization, a dissociation between decodability and use, or the discovery that the apparent loop is explained by sampling alone. The claim registry must narrow the paper accordingly.

Publish a minimal executable environment, source code, resolved configurations, split manifests, analysis-ready recordings, selected checkpoints needed for central claims, provenance, and figure source tables. Separate `replot` from `reanalyze` from `retrain`; rendering a figure from a CSV is not independent replication of the computation. The journal's policies require availability statements and access to central custom code for review, and identify reproducible public code release as best practice [N2].

## 13. First implementation milestone

The first vertical slice should load a real legacy model, replay paired 18 x 18 trajectories, compute a validated observer benchmark, record complete recurrent states, run leakage-resistant context/spatial decoding, compare initialization and trained checkpoints where available, and write one provenance-linked report. No active RL is needed for this milestone.

The next slice adds the transient cue, forced delay, final two-choice task, and recurrent task-learning loop. Only once this assay can distinguish unavailable evidence, recoverable-but-unused evidence, and actual task success should it be expanded to free navigation.

A “finished” infrastructure stage means the relevant invariants and end-to-end outputs have been executed and verified. A “supported” scientific stage means its registered comparisons and controls have produced evidence. These statuses must stay separate.

See `IMPLEMENTATION_PLAN.md` for module boundaries, task order, acceptance tests, and software contracts; `EXPERIMENT_REGISTRY.yaml` for the proposed experiment graph; and `SOURCES.md` for source locations and provenance.


---

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


---

# Sources and evidence provenance

Access date for web sources: 15 October 2026. Source-derived descriptions are distinguished from proposed designs in the research specification. No source below establishes the proposed new hypotheses. No scientific simulation or repository test was executed while preparing this package.

## [M] User-supplied manuscript

Manas Venkata Sai Ravulapalli. *Predictive learning folds spatial maps along symmetries*. Manuscript dated 6 October 2026. User upload: `main(20261015-105045).pdf`, 21 pages.

Relevant locations: model and training on pp. 14–16; exact indistinguishability and decoder caveats on pp. 15–16; transient room retention on pp. 5–9 and 19–20; Figure 5 on p. 9; population geometry in Figure 4 on p. 7; frozen-readout tasks on pp. 8–9 and 17–18; acknowledged biological limits on p. 13; rendering/graded-cue details on p. 19; source availability on p. 21.

These are results reported by the manuscript, not independently replicated in this package. The manuscript's parameters must be reconciled with source revision and checkpoint metadata during migration.

## [R] Public project repository

`RavulapalliManas/Symmetry_Cognitive_Maps`, public README and top-level listing. The older `Thesis_work` path redirected there when inspected. No source revision was pinned and no test suite was run in this inspection.

```text
https://github.com/RavulapalliManas/Symmetry_Cognitive_Maps
```

## [L] Navigation/context remapping comparator

Isabel I. C. Low, Lisa M. Giocomo, and Alex H. Williams. *Remapping in a recurrent neural network model of navigation and context inference*. eLife 12:RP86943 (2023), article identifier 86943.

Used for the established precedent of navigation/context-inference RNNs and aligned spatial maps. The proposed learning, active-sensing, and scaling experiments are not attributed to this study. Publisher pages intermittently returned a browser challenge; bibliographic and scope information was available in indexed primary content and the authors' repository description.

```text
https://elifesciences.org/articles/86943
https://pmc.ncbi.nlm.nih.gov/articles/PMC10328512/
https://github.com/ahwillia/rnn_remapping_paper
```

## [C] Behaviour–representation co-development comparator

Ian Cone and Claudia Clopath. *Latent representations in hippocampal network model co-evolve with behavioral exploration of task structure*. Nature Communications 15, 687 (2024). DOI: 10.1038/s41467-024-44871-6.

Used for the prior example of a plateau-based representation-learning model coupled to an agent's task learning. Their modelling assumptions and original task must be retained in any faithful reproduction; a changed implementation or environment is an adaptation.

```text
https://www.nature.com/articles/s41467-024-44871-6
```

## [B] Population abstraction analysis

Silvia Bernardi and colleagues. *The Geometry of Abstraction in the Hippocampus and Prefrontal Cortex*. Cell 183(4), 954–967.e21 (2020). DOI: 10.1016/j.cell.2020.09.031.

Used for the operational distinction between decoding and generalization of a decoding rule across conditions. The proposed spatial and developmental analysis design is an extension, not a result from this paper.

```text
https://pmc.ncbi.nlm.nih.gov/articles/PMC8451959/
```

## [PPO] Task-learning algorithm reference

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. *Proximal Policy Optimization Algorithms*. arXiv:1707.06347 (2017).

PPO is a proposed engineering choice for the task-learning component. The sequence handling, burn-in, loss routing, and legacy-model integration described in this package are implementation requirements to develop and validate; they are not claimed as directly supplied by this paper.

```text
https://arxiv.org/abs/1707.06347
```

## [N1] Journal scope

Nature Neuroscience, *Aims & Scope*. The journal includes computational modelling and prioritizes fundamental insight into nervous-system function. This establishes scope, not likely acceptance of any specific study.

```text
https://www.nature.com/neuro/aims
```

## [N2] Code, data, and reporting requirements

Nature Neuroscience, *Reporting standards and availability of data, materials, code and protocols*. Used for availability statements, reviewer access to central custom code, and reproducible release planning. Consult the policy again at submission; the package does not reproduce the full policy.

```text
https://www.nature.com/neuro/editorial-policies/reporting-standards
```
