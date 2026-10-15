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
