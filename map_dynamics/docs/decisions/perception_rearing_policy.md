# Perception, rearing, and state-dependent policy learning

Decision date: 17 October 2026. Status: accepted study direction; scientific hypotheses remain untested.

This is the maintained design record for the discussion of cue perception, developmental experience, and policy feedback.
The audience is the project owner and implementation contributors. The supplied research specification remains immutable.
The README owns current implementation status. `registries/preparation.yaml` owns experiment amendments, source evidence, run pointers, and unresolved requirements.

## Scientific question

How do sensory experience and behavioral demands jointly shape a predictive spatial representation?
The original question remains when a learner shares spatial structure, preserves contextual distinctions, or checks the environment again.
The rearing arm asks whether earlier experience changes those outcomes after the learner enters a common environment.
The policy arm asks how internal needs and external sensory evidence change behavior and the experience available for subsequent learning.
Both arms remain in the program. Neither requires a complete simulated visual cortex or mandatory animal-data acquisition.

The relevant distinctions are:

1. Evidence available in the permitted sensory and movement history.
2. Cue information accessible in visual features immediately after exposure.
3. Information accessible in recurrent activity during the delay.
4. Information used by the current policy.
5. How the resulting actions change future evidence and weight learning.

These are operational measurements, not an assertion that biological perception and memory occupy independent serial modules.
Successful decoding does not establish behavioral use. Failed decoding does not establish that all information is absent.

## Architecture and learning factors

The reference remains the preserved pRNN cell driven directly by local RGB and declared arrival signals.
A comparison adds a framewise visual encoder before the same recurrent cell and prediction, policy, and value heads.
The encoder receives the complete frame; it receives no cue label, context, coordinate, route ID, or Bayes belief.
The initial encoder comparison has no additional recurrent state, pooling, batch normalization, or pretrained weights.
It distinguishes an identity pathway, a frozen random convolutional encoder, and a trainable convolutional encoder.
Frozen random features are a control for a fixed transformation, not a model of mature visual cortex.
Pretrained visual encoders and recurrent sensory feedback remain later mechanism-specific comparisons.

Keep these factors separate:

| Factor | Main distinction |
|---|---|
| Sensory access | Which visual, motion, and later nonvisual signals are available |
| Sensory encoding | Identity, fixed features, or plastic features |
| Memory architecture | pRNN first; GRU/LSTM, Transformer, Mamba, and structured models later |
| Objective | Sensory prediction, task return, or a declared combination |
| Credit assignment | BPTT and explicitly derived online/local alternatives |
| Optimizer | Adam, SGD, or the preserved legacy schedule |
| Experience | Imposed histories, active sampling, rearing order, and offline replay |

RFLO and e-prop are computational learning algorithms motivated by locality and eligibility traces.
BTSP and STDP refer to biological plasticity mechanisms and their explicit computational models.
Replacing Adam does not implement a biological plasticity rule.
The layer-normalized legacy cell couples units; a local rule requires a derivation for that cell.
Changing the cell, normalization, objective, and rule together cannot isolate an update-rule effect.

## Cue perception and environmental variables

Cue validity is the probability that a displayed symbol identifies the true context.
Cue contrast controls its rendered appearance. Duration controls exposure, not independent evidence count.
Repeated copies of one noisy symbol remain one evidence event.

| Variable | Required separation or control |
|---|---|
| Visibility | Contrast, size, occlusion, and clutter versus cue validity |
| Discriminability | Immediate cue-identity readout versus delayed context readout |
| Recognition across views | Same cue under position, viewpoint, texture, and lighting changes |
| Sensory redundancy | Different locations producing the same current observation |
| Sensorimotor correlations | Cue identity, heading, movement, and context correlations |
| Predictive relevance | Whether a distinction changes targets within the trained horizon |
| Exposure | Dwell, viewing frequency, speed, and repeated versus independent samples |
| Learning history | Prior environments, presentation order, and common later adaptation |
| Sensory conflict | Visual evidence against motion or another declared modality |
| Visual complexity | Clutter and predictability separately from cue information and task difficulty |

Naturalness is not defined by one image-complexity score. NCA-generated images are an optional renderer, not a necessary environment family.
Changing background predictability can change raw-pixel prediction demands without changing cue truthfulness.
The initial fixed-pixel cue remains a calibration signal. It does not establish landmark recognition or natural vision.
Scene-embedded cues and controlled appearance changes follow a working perception assay.

The first implementation changes contrast without adding pixel noise.
For any representable nonzero contrast, an exact observer can still identify the displayed symbol.
Its posterior therefore depends on cue validity, not contrast magnitude.
At zero contrast, symbol identities coincide and the context posterior remains the prior.
Positive contrast that collapses under float32 storage must fail rather than silently become zero contrast.
This experiment tests learner sensitivity to rendering; it does not claim a calibrated biological detection threshold.

Diagnostic cue classifiers fit on fitting families only and never backpropagate into the agent.
Record the presented symbol separately from true context in analyst metadata.
Measure raw input and encoder cue access, post-cue recurrent context access, delayed access, and task behavior.
Equal immediate classification accuracy does not imply identical internal representations.
A separately trained immediate-response capacity control remains necessary before attributing a scientific failure to forgetting.

## Rearing and policy feedback

The rearing bridge changes experienced statistics and then evaluates all groups on common histories.
Candidate manipulations include cue-heading correlations, visual redundancy, route structure, and cue availability.
Vary one declared factor first. Terms such as lab-like and wild-like must resolve to these parameters.
They are not evidence that laboratory and wild animals differ in the predicted way.
Match total experience and use a common final adaptation phase. Separate optimizer continuation from reset.
Report history dependence unless recovery and alternative controls justify a stronger critical-period or hysteresis claim.

Internal body state, such as modeled energy, is distinct from recurrent memory state.
The planned policy reads the predictive state and declared body variables, not the true external world state.
The predictive-state variants withhold body state, provide it as input, or additionally predict it.
These are distinct input/objective experiments. Reward timing and body observations must enter the observer's information contract.

The reference closed loop lets the map learn prediction while the policy learns its declared reward.
Stop policy gradients at the map to isolate indirect effects through sampled experience.
Joint task gradients are a separate comparison.
Intrinsic prediction-error reward, external reward, and homeostatic reward remain separate declared conditions.
Prediction error must not automatically be interpreted as information gain; unpredictable noise is a competing source of error.

Distinguish immediate action changes under fixed weights from policy learning and representational learning.
Use frozen-map, frozen-policy, standardized-history, and cross-yoked predictive-learning controls where they identify the stated contrast.
Do not apply uncorrected on-policy PPO to another policy's replayed actions.
Future movement used by a conditional prediction branch must never enter the live action-selection state.
Prediction under recorded actions does not by itself establish correct counterfactual planning under arbitrary actions.

## Ordered experiment amendments

The original X0-X5 program and untested claims remain intact. Amendments do not create an exhaustive Cartesian sweep.

| ID | Question and comparison | Endpoint and failure interpretation | Dependencies |
|---|---|---|---|
| X1-P | Does visual processing change immediate cue access and later retention? Identity, frozen random CNN, trainable CNN; contrast null and nonzero controls | Cue-symbol access, context access, policy accuracy; an encoder effect does not establish a V1 mechanism | Existing X1; immediate-response capacity control before scientific interpretation |
| X2-D | Does context disappear, recode, suffer interference, or remain unused? Matched histories and targeted versus control perturbations | Held-out response prediction and fixed/refitted readouts; clustering alone cannot establish attractors | X1-P calibration and original X2 |
| X4-R | Does rearing change later learning after current evidence is matched? Different cue-motion statistics followed by common tests and adaptation | Initial cue access, retention, adaptation cost, interference; a difference explained by current sensory difficulty is not an isolated memory effect | X1-P and original X4 controls |
| X3-S | How do internal needs and external evidence change policy and subsequent maps? Frozen and adaptive maps/policies, declared state access | Acute choices, experienced evidence, standardized retention, later weight changes; changed routes alone do not identify improved memory | Active environment, state contract, valid observer, original X3 |
| X2-RP | Does learned structure constrain offline replay? Decode online and offline activity with ambiguity-aware readouts | Coherence, content, and direction controls; undecodable room identity blocks room-specific replay claims | Verified online code and original dynamics tests |
| X5-T | Can pre-transfer measurements predict new-environment learning and compensation? Freeze predictions before held-out layouts | Transfer, adaptation, and interference against cue-age, route, and size baselines | Working mechanism from preceding experiments |

Architecture families, multisensory perturbations, learning rules, and organism comparisons are selected replications after the first interpretable result.
They are retained directions, not claims of implementation or a commitment to run every combination.
Animal comparisons require a species, accessible dataset, and a matched observable; no animal acquisition is a dependency.
The existing dataset shortlist in `preparation.yaml` remains the owner of those candidates.

## Neuroscience and novelty boundaries

Levenstein and colleagues supply the sequential-prediction and replay reference.
Their discussion identifies prior experience, multi-environment learning, and cortical hierarchy as future directions.
Du and colleagues supply a direct precedent for sensory-prediction-error-driven exploration.
The verified McGill item is a showcase proceedings abstract, accompanied by a poster; these are not independent experiments.
An older local review that failed to identify Sabrina Du is superseded by the matching primary publication.
Sun and colleagues already study spatial-map learning trajectories and compare model families.
Madar and colleagues already compare plasticity models against measured place-field shifts.
Input sparsity interacting with memory architecture is also quantified by Lin and colleagues.

Neither adding an encoder nor coupling prediction to exploration is a novelty claim.
The candidate contribution is a prospective mechanism linking controlled sensory evidence and experience history to map differentiation, retention, policy compensation, and transfer.
Existing work must be cited when an endpoint overlaps it. No exhaustive novelty claim is established by this review.

Primary sources and the exact supporting excerpts are maintained in `discussion_checkpoint.sources` in `registries/preparation.yaml`.
The reference links are [Levenstein](https://doi.org/10.1101/2024.04.28.591528),
[Du](https://doi.org/10.26443/msurj.v21i2.437),
[Sun](https://doi.org/10.1038/s41586-024-08548-w),
[Madar](https://doi.org/10.1038/s41593-025-01894-6),
[Fiser](https://pubmed.ncbi.nlm.nih.gov/27618309/),
and [Lin](https://arxiv.org/abs/2510.09951v3).

## Implementation boundary and migration

The immediate increment is X1-P software development. It keeps the existing development caps and raw RGB prediction targets.
The encoder is framewise and preserves the backbone input width; its initialization must not change paired backbone, head, or rollout-noise initialization.
Encoder parameters, task contrast, source fingerprints, and optimizer/RNG state belong in the checkpoint contract.
Changed source or protocol rejects old continuation rather than silently reinterpreting a checkpoint.
The pinned `legacy_reference/` remains unchanged.

Development runs verify execution, gradient routing, calibration nulls, and artifacts. They cannot support the registered scientific hypotheses.
The full pilot still needs a frozen protocol, capacity controls, independent layouts, replicate planning, hard compute budget, and durable archive.
An available L40S is hardware availability, not an approved GPU-hour budget.
