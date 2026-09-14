# Melody Matcher

I think Spotify and Apple Music recommendations are pretty shit. I keep discovering songs I absolutely love that have been sitting there for decades. They didn't need to be written. They needed to be recommended to me.

I don't know why I love both HARDY's "Favorite Country Song" and Aretha Franklin's "Don't Play That Song," but I'm pretty confident there's something in the music's **DNA** that explains it. Not a genre label. Not a "danceability" score. Something my brain hears even when I can't put it into words.

A friend who knows your taste can hear a song and think, *you need to hear this*. They've built a mental model of what hits for you. I believe we can train a neural network to learn those connections from the music itself and the music people actually return to, without first teaching it our vocabulary for describing songs. That's the bet behind Melody Matcher.

The ambition goes beyond another playlist. Sometimes I'm obsessed with one song and want something that scratches the same itch before I play it to death. Sometimes I want a song from somewhere I'd never have thought to look. A good model of musical DNA should let us explore both. [The thesis and the reasoning](docs/RESEARCH.md) explain what I mean, what we tried, and where we got stuck.

I also think the same logic could reach far beyond music: books, movies, maybe even clothes. Represent the thing richly, then train a large neural network to learn how people's preferences relate to it. Music is the first place we're trying to make that work, not the limit of the idea.

**The thesis is alive. The project is paused, and the recommender doesn't work well.** Our current suspicion is that we don't have the right listening data, or enough of it, to train the model properly. Making the network bigger without solving that doesn't get us anywhere. We're sharing the source and thinking so somebody else can work from them, not because we're pretending to have solved music discovery.

This is an opinionated research project, not a supported product. The source uses audio embeddings and candidate-conditioned attention over a listener's library. The historical trials, limitations, and possible next experiments are documented below; no new benchmark or runtime validation was performed for this release.

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
