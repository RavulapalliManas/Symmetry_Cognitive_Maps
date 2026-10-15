# Learning when to share a cognitive map

A research-and-implementation specification for a computational-only study of recurrent population coding, contextual memory, active evidence-seeking, and generalization. The intended journal target is Nature Neuroscience.

**This is a design package, not a working experimental repository.** It contains no trained models, newly measured results, executed scientific tests, or modifications to the public project repository. The YAML files describe proposed experiments and hypotheses; they are not executable training configurations.

## Contents

| File | Purpose |
|---|---|
| `RESEARCH_SPEC.md` | Scientific question, inherited evidence, model/world specification, analyses, experiments X0–X5, causal tests, statistics, and interpretation |
| `IMPLEMENTATION_PLAN.md` | Proposed package structure, interfaces, data contracts, test families, milestones M0–M6, and definition of the first end-to-end implementation |
| `EXPERIMENT_REGISTRY.yaml` | Machine-readable experiment dependencies, pilot design, control requirements, and unimplemented status |
| `CLAIMS.yaml` | Hypotheses, required evidence, potential narrower outcomes, and explicitly untested status |
| `AGENTS.md` | Development instructions for human contributors and coding agents |
| `SOURCES.md` | Manuscript locations, primary literature, public-repository inspection boundary, and journal-policy sources |
| `PACKAGE_MANIFEST.json` | SHA-256 checksums, source-upload fingerprint, and package status |

## Start here

Read the project definition in `RESEARCH_SPEC.md`, then implement M0–M1 from `IMPLEMENTATION_PLAN.md`: preserve the legacy code, audit the actual assets, load one real model, validate the information available on paired trajectories, record its population, and produce one leakage-resistant source-linked analysis. Active RL and larger arenas come after that vertical slice works.

The early controlled-learning pilot proposes eighteen networks: two task requirements x three learning modes x three paired seeds on one development layout. This is a feasibility pilot, not a power calculation or a confirmatory campaign. Positive controls and further experiments are budgeted separately.

The main scientific hypotheses remain untested. Their rejection should narrow the claim rather than trigger undocumented adjustments designed to recover the desired story.
