# A possible next proof, not the next product

[July reset in the chronology](HISTORY.md#8-may-and-july-reconsider-the-evidence-and-the-scale-of-the-test) · [Existing training targets](TRAINING_EXPERIMENTS.md) · [Experiment index](EXPERIMENTS.md)

Moritz's July 13 proposal was to return to the originating thesis: enough clean behavior, rich audio representations, attention, and enough model capacity to learn connections without hand-authoring what taste should mean. Stop optimizing the imagined product interaction and find the fastest credible test.

The benchmark below is Marty's proposed response. It was not adopted or executed. No retained trainer implements it, and no data, results, or new collection permission are supplied.

## Keep the evidence closer to what was observed

The existing model compresses aggregate plays into normalized scores. That creates a convenient target, but makes choices about what constitutes strong preference before learning begins.

Moritz proposed using embeddings and actual play behavior, then experimenting with additional song and user metadata: time since addition, duration, release age, and other available context. Let the model learn which information helps. Explicit ratings from every listener were not the desired foundation because the input collection itself needs to scale.

Raw behavior is still not pure preference. Plays depend on exposure, time available, activity, playlists, habit, and access as well as affection. A larger network can learn those confounds more effectively too. The language-model analogy motivates the bet; it does not establish that missing exposure information will resolve itself with scale.

## First ask whether someone comes back

A narrower first question is:

> After a listener has encountered a song, can its audio and their prior history help predict whether they return to it, and how much they replay it?

An observed play reduces the worst unseen-versus-rejected ambiguity. It does not prove this was their first encounter, that they selected it deliberately, or that later plays reflect affection rather than playlist placement.

The proposed outputs are:

1. Return within a fixed future window.
2. Future count or rate conditional on return.

This is different from the existing hurdle model's prepared-library membership and transformed aggregate-score heads. A count model would need its own likelihood and exposure/time interpretation. Poisson and an over-dispersed alternative such as negative binomial were suggested baselines, not supplied implementations.

## Define a data contract before fitting anything

A continuation would need to specify:

- **Observation coverage:** which events are recorded and which devices, periods, or services are missing.
- **Cutoff:** the moment separating inputs from outcomes.
- **Horizon:** the fixed future window in which return and replay are measured.
- **Exposure definition:** what qualifies as a known encounter and how uncertain first encounters are handled.
- **Censoring:** what happens when observation ends early, a listener disappears, or a song becomes unavailable.
- **Feature timing:** counts, recency, popularity, and metadata must be computed as known at the cutoff, not using future information.
- **Splits:** time-based holdouts plus separately defined unseen-listener and cold-song cases.
- **Rights:** an authorized source of events and audio representations, consent or another applicable basis, retention rules, and separate redistribution decisions.

The [event collector](../sequential/README.md) is possible infrastructure, not proof that enough usable temporal history exists. Saved libraries and all-time top tracks cannot silently substitute for an event record with known observation coverage.

## Make the audio earn its place

Use one frozen task and evaluation construction to compare:

- popularity, listener activity, prior plays, and availability/age;
- collaborative behavior without audio;
- pooled audio similarity and a regularized linear predictor;
- metadata-only neural prediction;
- audio plus behavior;
- audio plus behavior plus metadata;
- a shuffled-audio control that breaks the correct song/audio pairing while preserving other information.

A gain over weak baselines would not establish the interesting claim. The question is whether the musical representation adds predictive value beyond behavior and popularity, especially on less-popular songs and cross-category cases. Shuffling and preprocessing must remain inside the evaluation boundaries so the control does not introduce leakage of its own.

Report return calibration and conditional-count error separately, then relate them to actual listener satisfaction. A model can predict habitual replay without finding a new favorite.

## Scale data and model together

The current model might be too large for the available evidence while still being far too limited for the broader ambition. Those are compatible hypotheses.

A useful test would hold the task fixed and vary both event volume and model capacity within one architecture family, using repeated seeds and recording compute. Look for held-out improvement, not only lower training loss. More parameters cannot compensate for an undefined target or missing evidence by definition; neither does a small failed run establish that the larger thesis is false.

There is no controlled scaling curve in this archive. The external encoder's MERT-v1-330M name is also not the parameter count of the downstream preference model.

## What this would prove—and what it would not

If audio-aware models reliably improve prediction of return and replay under these controls, that would support a narrower claim about preference-related behavior after exposure. It would not yet prove discovery of never-before-played songs.

Discovery requires a defensible account of what candidates the listener could have encountered and a listening evaluation that distinguishes familiarity from satisfaction. A blinded protocol with consented participants could complement the numerical task. The April lesson still applies: a favorable number and music someone actually loves are different outcomes.

This is a proposed route to better evidence, not a requirement that anyone adopt the last architecture. The archive is useful if it saves the next researcher from repeating our assumptions—even if their next experiment uses a different representation, a simpler model, or a better data source.
