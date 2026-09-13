# Research

This account separates **implemented behavior** established by static source reading, **historically reported observations**, **interpretation**, and **proposals**. Source presence is not evidence that an experiment completed successfully. The snapshot includes neither the original data nor the artifacts needed to reproduce its historical results.

## Question and hypothesis evolution

The originating hypothesis was that musical preference involves relationships more specific than genre membership or global audio similarity. A listener may connect otherwise dissimilar songs through a small set of musical characteristics. Audio representations and behavioral evidence might allow a model to discover those connections without explicitly prescribing musical rules.

Early descriptions used a two-tower framing. The retained implementation instead lets each candidate attend to a library of track representations. This avoids forcing every candidate to use the same single pooled user vector. It does not establish that learned attention identifies meaningful musical properties.

The **dimensional-selectivity hypothesis** is that preference depends on different aspects of audio for different listeners and candidates. FiLM and attention provide mechanisms that could express such selectivity. Individual embedding coordinates are not demonstrated to correspond to named musical traits, attention weights are not causal explanations, and an expressive mechanism is not proof that the intended relationship was learned.

Human choices remain throughout the process: artist seeds, genre/tag mappings, diversity thresholds, overlap gates, track filtering, target transforms, and candidate construction. The scorer does not directly apply genre rules, but the overall research pipeline is not assumption-free.

## Implemented architecture

The authoritative local definitions are [model.py](../library/src/model.py), [layers.py](../library/src/layers.py), and [config.py](../config.py).

1. The selected external representation is MERT-v1-330M, configured for **1024-dimensional embeddings** and 24 kHz audio. The [extractor](../src/embedder.py) mean-pools the last hidden state over time and L2-normalizes it. Its remaining 768-dimensional comments are stale. Model weights are absent.
2. `ScoreFiLM` maps each scalar context score through separate scale and shift networks, each with a 64-unit hidden layer. It applies `gamma(score) * embedding + beta(score)` before the library projection.
3. The library encoder projects to a default width of 512 and uses three self-attention layers, eight heads, and feed-forward width 2048. A context track's score is added to its key attention logit across queries and heads. FiLM conditioning and this attention bias are distinct mechanisms.
4. Candidates are projected to the same hidden width and pass through two cross-attention layers over the encoded library. There is no candidate self-attention in these decoder layers. A regression head maps each candidate feature vector through width 256 to one scalar.
5. The hurdle variant shares the encoder and candidate backbone, with separate play-logit and score heads. Both are small output MLPs. The class defaults are not a statement of every historical trial's configuration.

There is no positional encoding in the retained library encoder. Static structure supports a set-like reading: jointly permuting context embeddings, scores, and masks should preserve a candidate's deterministic evaluation output, subject to numerical effects. That property has not been tested here. The encoded context itself is a sequence of vectors whose order follows the input; the claim concerns the final candidate scores.

Chronological event collection in [sequential/scripts](../sequential/scripts) does not implement a sequential recommender. Those timestamps are not fed into the retained aggregate-library model.

## Implemented targets and evaluation

[Preprocessing](../library/src/preprocessing.py) transforms aggregate play counts into per-user targets: log-normalized counts, rank percentiles, a top-quartile indicator, or a clipped log-count z-score. The default log transform divides each track's `log(1 + count)` by the maximum within that prepared library. These are constructed preference proxies, not direct measurements of liking.

[TasteDataset](../library/src/dataset.py) samples up to 300 tracks by default, randomly permutes them, and divides the window into 70% context and remaining targets. Users are assigned by `user_id % 10` to an 8/1/1 train/validation/test split. Neither split is temporal evaluation. Scores are constructed before the context/target division, so even the normalization includes the held-out portion of that prepared library.

Optional negatives are sampled from the embedding catalog while rejecting tracks in the user's **prepared** library. Filtering can already have removed known tracks from that library. These negatives therefore are not observed rejections or necessarily unplayed songs. The sampler permits repeated negative draws.

The existing hurdle objective uses binary cross-entropy for prepared-library membership versus sampled negatives, plus weighted mean squared error on positive transformed scores. Labels such as `P(played)` in the harness are historical shorthand. They do not establish calibrated discovery probabilities, known exposure, or future replay counts.

[Training metrics](../library/scripts/train.py) compute Spearman and AUC after concatenating up to the last ten training batches; validation/test concatenate all loader batches. Spearman uses only nonpadding positive targets, while AUC contrasts nonpadding positive and sampled-negative targets. For hurdle models, ranking metrics use `sigmoid(play_logits)`, not the regression head. Pooled Spearman is not mean per-listener correlation. NDCG is computed over constructed target sets. These are not a catalog-wide temporal discovery benchmark. Varying target transforms, cohort membership, negative ratios, and random windows can alter metric meaning and difficulty. A fixed training recipe alone does not make those comparisons causal.

## Historically reported result and failure

A development checkpoint dated **April 22, 2026** reported an 18-trial sweep, with best validation Spearman **0.719** for a longer-training trial. It also reported a listening inspection in which the top recommendations were unfamiliar and strongly clustered in a narrow, slow instrumental/background style, and were judged unsatisfactory.

This is a dated historical report, not an independently verified benchmark or a reproducible listening study. Trial logs, evaluated checkpoints, exact data/splits, recommendation lists, and listening judgments are not distributed. The current source has evolved since that report, including hurdle-related changes; its presence cannot recreate the historical execution state. The report's claims of a working harness and a settled diagnosis are not adopted as current guarantees.

The useful observation is the reported disagreement between an offline score and the listening experience. Unfamiliarity alone would not prove failure at discovery, but the reported dissatisfaction and narrow clustering motivate evaluation beyond that score. The single inspection does not establish population-wide recommendation quality.

## Competing interpretations

The April interpretation was **training-distribution collapse**: a narrow supervised signal could reward one audio region regardless of context. No controlled comparison establishes that as the cause. Other explanations remain plausible:

- Aggregate counts conflate exposure, availability, activity, habit, and preference; the objective may reward the wrong behavior.
- Within-library regression may extrapolate poorly to catalog discovery, while sampled negatives can introduce false negatives and sampling-prior effects.
- Coverage and fuzzy matching may distort the audio or library evidence presented to the model.
- Representation choice, optimization, capacity, implementation defects, or inference caching may affect results.
- Offline evaluation may reward correlations that do not improve listening satisfaction.

The retained dataset and negative-sampling experiments make several of these ideas concrete in code. They do not supply a demonstrated resolution. A linear baseline remained a research proposal; it is not presented as an implemented experiment here.

## July direction: return after known exposure

A **July 13, 2026 proposal** reframed the smallest useful proof: after a listener has encountered a song, predict whether they return within a future window and how intensely they replay it if they return. Inputs would use only pre-cutoff behavior and information available then, with count, recency, and metadata kept explicit rather than collapsed into a handcrafted score.

That proposal is different from the existing membership/score hurdle target. It requires an explicit time horizon, exposure definition, censoring policy, and time-correct features. A first observed play reduces exposure ambiguity but may not be the listener's true first encounter. Repeat behavior also remains affected by playlist placement and habit.

Temporal holdouts, strong behavioral baselines, shuffled-audio controls, and a model/data scaling ladder could test whether audio adds useful predictive information. More capacity is an empirical variable, not an established solution. [Experiments](EXPERIMENTS.md) develops these proposals into small distinguishing tests; none is promised as a development roadmap.
