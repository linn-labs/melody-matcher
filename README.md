# Melody Matcher

Can a model learn which aspects of a song matter to a listener from audio representations and listening behavior? Melody Matcher explores that question with candidate-conditioned attention over a listener's music library.

**This is unfinished, paused research, not a working or supported recommender.** The source is shared to make the approach, experiments, and unresolved questions understandable. Recommendation quality has not been validated. No installation, training run, harness session, or benchmark was reproduced for this snapshot.

The most instructive historical observation is a mismatch: an April 2026 record reported validation Spearman of 0.719, while listening inspection found narrowly clustered, unsatisfactory recommendations. That metric is an unverified historical report, and the cause of the failure remains unresolved. See [Research](docs/RESEARCH.md).

## Read the research

- [Research](docs/RESEARCH.md): hypothesis evolution, actual architecture, historical observations, competing explanations.
- [Experiments](docs/EXPERIMENTS.md): implemented avenues and proposed tests that could distinguish those explanations.
- [Components](docs/COMPONENTS.md): source navigation, expected schemas and artifacts, preexisting defects, and reader hazards.
- [Data and rights](docs/DATA_AND_RIGHTS.md): privacy and external model, music, and service restrictions.
- [Provenance](PROVENANCE.md): what this curated snapshot includes and excludes.

## Source map

| Path | Research material |
| --- | --- |
| [library/src](library/src) | Score-conditioned library encoder, candidate cross-attention, regression and hurdle heads, dataset construction |
| [library/scripts](library/scripts) | Historical preparation, training, dataset, negative-sampling, and hyperparameter experiments |
| [src](src) and [scripts](scripts) | Collection schema, service clients, sampling, matching, and embedding extraction |
| [sequential](sequential/README.md) | Timestamped listening-event collection; no sequential training model |
| [harness](harness/README.md) | Experimental library import, model comparison, ranking, and feedback source |
| [library_analysis](library_analysis/README.md) | Coverage comparison against a separately supplied library |
| [config.py](config.py), [requirements.txt](requirements.txt) | Historical configuration and dependency declarations |

No listening data, personal library exports, audio, embeddings, weights, checkpoints, databases, or experiment logs are supplied. The harness requires absent artifacts and is not a hosted demo. Its source contains unsafe artifact loaders, storage writes, and external browser requests. Read the component guide before considering execution; source examples are not tested setup instructions.

Owned source and documentation are licensed under [MIT](LICENSE), copyright Linn Autoracing Excellence LLC. That grant does not cover external models, music, service data, or derived artifacts. In particular, the selected MERT model has a noncommercial license; see [third-party notices](THIRD_PARTY_NOTICES.md).

Forks and research contributions are welcome. Active maintenance, timely review, and operational support are not promised. See [Contributing](CONTRIBUTING.md).
