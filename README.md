# Melody Matcher

Music recommendations should help us discover songs we love for reasons we can't quite explain. Melody Matcher is an attempt to learn those reasons from the music itself and people's listening behavior.

I love both HARDY's "Favorite Country Song" and Aretha Franklin's "Don't Play That Song," without being able to explain what connects them. That's the music's **DNA**: something the brain hears that a genre label or a "danceability" score fails to describe.

A friend who knows your taste can hear a song and immediately know you need to hear it. Train a neural network to learn those connections from the music itself and people's listening behavior, without first teaching it our vocabulary for describing songs. That's the bet behind Melody Matcher.

The ambition goes beyond another playlist. When you're obsessed with a song, find something that gives you the same feeling before you play it to death. Or reach further out and discover music you'd never have thought to look for. [The thesis and the reasoning](docs/RESEARCH.md) explain the idea, what we tried, and where we got stuck.

If this works, the same principle can extend to books, movies, maybe even clothes: represent the thing richly, then train a large neural network to learn how people's preferences relate to it. Music is the first test.

**The project is paused, and the recommender doesn't work well.** The current hypothesis is that we lack the right listening data, or enough of it, to train the model properly. Scaling the network without scaling that evidence doesn't get us anywhere. We're sharing the source and thinking so someone else can work from them.

The source uses audio embeddings and candidate-conditioned attention over a listener's library. The docs cover the historical trials, limitations, and possible next experiments. This is unfinished research; the experiments weren't rerun for this release.

*In these docs, ‘I’ is Moritz Linn; ‘we’ means Moritz and his AI collaborators. Marty is Moritz's custom AI agent built on [Hermes](https://github.com/NousResearch/hermes-agent), one of those collaborators.*

## Read the research

- [The thesis](docs/RESEARCH.md): music DNA, what better recommendations would make possible, and why the project is paused.
- **[The research history](docs/HISTORY.md): what we believed → what we built and tried → what happened → what we realized → what changed next. Start here to learn from the attempts.**
- [Experiment index](docs/EXPERIMENTS.md): the chronological route into detailed sampling, training, and future-test explanations.
- [How the snowball works](docs/SAMPLING.md): why the collection plan changed, exactly what overlap means, and what each cohort was meant to test.
- [Training experiments](docs/TRAINING_EXPERIMENTS.md): the original regression task, all dataset-variant families, negative sampling, hurdle heads, and evaluation limitations.
- [A possible next proof](docs/FUTURE_PROOF.md): the later exposure/return proposal, not an implemented experiment.
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
