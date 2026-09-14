# Research

This account separates **implemented behavior** established by static source reading, **historically reported observations**, **interpretation**, and **proposals**. Source presence is not evidence that an experiment completed successfully. The snapshot includes neither the original data nor the artifacts needed to reproduce its historical results.

## The listener problem and the thesis

Mo's starting point was dissatisfaction with recommendations that missed personally meaningful connections between otherwise dissimilar songs. The desired discovery was a song that felt right for a particular listener even when its genre, instrumentation, or overall sound differed from familiar favorites. The research question is what evidence a model would need to learn that relationship.

Broad categories can describe a collection without explaining which songs within it resonate. A single global audio-distance rule can emphasize similarities that matter little to one listener and overlook those that matter greatly. Co-listening can reveal useful associations, but observed plays also reflect what people encountered and had opportunities to replay. These are limitations to investigate when relying on those signals, not a description of undocumented commercial internals. Modern recommenders can combine behavioral, audio, and other signals; the project's dissatisfaction is neither an industry-wide evaluation nor a novelty or superiority result.

The thesis is ambitious: let a sufficiently capable model learn musical relationships from audio and scalable behavioral evidence rather than require a human vocabulary of musical rules. A candidate should be able to draw on different parts of the same listener's history. Attention interested the project because it offers that flexibility; scale interested Mo because a small model on sparse evidence might never have enough capacity or examples to discover the relationships being sought.

The **dimensional-selectivity hypothesis**, made explicit in the April design records, is that preference depends on different aspects or combinations of audio features for different listeners and candidates. This is a hypothesis about learnable structure, not a claim that particular embedding coordinates have established musical meanings. Attention weights are not causal explanations, and an expressive mechanism does not prove that the intended relationship was learned.

Human choices remain throughout: artist seeds, genre/tag mappings, diversity thresholds, overlap gates, track filtering, target transforms, and architecture. The scorer does not directly apply genre rules, but the overall research pipeline is not assumption-free. The history below shows how attempts to make the thesis testable introduced assumptions of their own.

## How the approach took shape

### Early plan and March: obtain behavioral evidence and usable audio

The initial plan paired a pretrained audio representation with Last.fm track counts, then proposed a two-tower model that would summarize a listener into one vector for nearest-neighbor retrieval. Its rationale for counts was practical: track-level listening records offered repeated behavioral evidence without asking every listener to rate every song. Counts were treated as a starting proxy for preference, not a direct observation of why a song mattered.

That plan considered MERT, CLAP, OpenL3, and Jukebox representations and recommended starting with MERT for its music-specific focus and a tractable representation size. This was a design judgment, not a project comparison demonstrating that MERT best captured personal taste. The retained extractor uses MERT; the exact dimensions below supersede stale numbers in the early plan. Using a pretrained representation made the downstream taste question approachable without first training an audio encoder, while leaving open whether its features retained the information the task needed.

The **March 3, 2026 checkpoint** described discovery aimed at listeners with broad tastes, followed by aggregate top-track collection and diversity filtering. The intention was to expose the model to connections across musical categories. The same record reported uneven discovery yield and a strong skew toward active listeners with deep histories. It recorded a decision to proceed with the collected pool rather than immediately repeat failed discovery attempts, pending the diversity results. Coverage was already a research constraint, not just a count of collected users.

The **March 5 checkpoint** separated two questions: durable taste from unordered libraries, and next-song context from chronological events. It described Deezer preview matching and timestamped collection infrastructure. The **March 24 checkpoint** then reported that saved preview URLs had expired before embedding extraction; matching and extraction were consolidated so previews could be resolved near use. That was a response to an acquisition failure, not evidence of a better taste representation. These dated reports do not establish current service availability or permission; [Data and rights](DATA_AND_RIGHTS.md) covers that separate boundary.

### April design: score each candidate against the library

The **April 7 principles and April 9 scoring design** moved away from the initial single-vector framing. The stated concern was that averaging a diverse library into one point, even with one set of dimension weights, could obscure multiple ways in which songs appeal to the same person. The chosen design kept a library of representations and let each candidate attend to it. This is a documented architectural rationale, not a measured defeat of all pooled-vector or two-tower alternatives.

The April 9 design also explained why scores entered through FiLM and an attention bias. A play-derived score describes a listener's relationship to a track, so it was intended to control how the audio representation was used. Concatenation was considered; uniform multiplication was considered too. FiLM offered learned scale and shift by dimension, while the separate attention bias increased the influence of higher-scored context tracks. Both mechanisms are present in the retained code. Their benefit over simpler conditioning remains unestablished without ablations.

Within-library regression made a first task possible: reveal part of a library and predict transformed play counts for the remainder. The design preferred log normalization to compress large count differences while preserving ordering. It considered alternatives because old songs, background listening, and listener activity complicate count interpretation. The retained preprocessing and dataset-variant code implements several transforms and filters; their presence does not establish a winning target. An April 9 per-listener linear-regression baseline was also proposed as a cheaper probe of whether simple feature weights carried signal. It remained a proposal, not a completed comparison or proof that nonlinear attention was necessary.

The same scoring design initially rejected sampled negatives: a song absent from a library might simply be unheard. It hoped relative scores among known tracks would be enough for ranking. That choice avoided labeling unknown songs as dislikes, but left an unanswered extrapolation question when the trained scorer was applied to the wider catalog.

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

The April 22 record explains why a listening harness was added after the training sweep: predicting held-out scores did not answer whether the model's actual catalog recommendations were useful. Import, comparison, and feedback were intended to connect the offline task to that experience.

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

## After the listening mismatch: change the sample, or change the task?

The **April 22 retraining plan** prioritized changing the training distribution. Its reasoning was that catalog overlap did not establish useful supervision: finding familiar tracks in the embedding catalog did not show that training examples taught the model when to score related candidates highly. It considered inference-time filtering, reweighting existing examples, and collecting a more relevant cohort. The plan favored new collection, with reweighting as an intermediate diagnostic. That was a response to the sampling diagnosis described above; it did not experimentally rule out capacity, objective, or implementation problems.

The **April 27 snowball plan** recorded a further decision, informed by a spot check: replace one uniform overlap filter with multiple tagged cohorts. The record argued that a popularity cap could exclude relevant musical regions and that modest overlap with a diverse library did not itself guarantee a broadly similar listener. It proposed stricter breadth-oriented overlap, looser region-filling collection, friend expansion, and long-tail seeds. Tagging cohorts would allow later training subsets to test different sampling theories without repeating collection. The plan records acceptance of that upfront collection tradeoff. The retained snowball and dataset-variant code supports these mechanisms; it does not supply a controlled result showing which cohort helped. In particular, stricter overlap was intended to encourage breadth, not a demonstrated guarantee of it.

A **later negative-sampling and hurdle plan**, whose body has no reliable experiment date, revisited the April rejection of negatives. It diagnosed a different gap: the scorer had only seen targets already belonging to a listener's library. It proposed a fixed dataset and backbone, varying the number of random catalog targets to isolate the loss/target change. The hurdle variant separated membership discrimination from positive-score regression, intended to avoid a single MSE objective being dominated by many zero targets. Those variants are implemented in the retained source, but no distributed sweep artifacts establish an outcome or an adopted winning configuration. Their classification labels still mean prepared-library membership versus sampled absence, not known exposure or dislike.

These threads identify different uncertainties. Broader coverage might improve the available evidence; targeted cohorts might increase relevant supervision; new losses might change what the scorer learns to distinguish. None alone resolves whether the evaluation target captures satisfying discovery. The negative-sampling concern did not disappear when negatives were implemented: it became a tradeoff to measure. [Experiments](EXPERIMENTS.md) separates those mechanisms and proposes comparisons that could distinguish them.

## May and July: scale the thesis, first define the proof

On **May 20, 2026**, Mo questioned whether a model at the project's data and parameter scale could meaningfully test the thesis. His analogy was an extremely small language model: weak output might reflect insufficient capacity and evidence rather than the absence of learnable structure. He also argued that arbitrary track order should not matter for aggregate play-count modeling, while real event order could matter for listening-history questions. This explains the interest in attention without arbitrary ordering effects; it does not establish that the May discussion caused the already documented April architecture. The retained set-like model and chronological collector remain distinct.

In his **July 13 note**, Mo returned to the fastest way to prove the originating idea, setting aside product choices such as library import versus individual song selection. Scalable inputs remained central: explicit song ratings would require manual judgment at a scale the project could not assume. He proposed treating embeddings and observed counts as evidence, trying song/user metadata as additional inputs, and letting a capable model learn what mattered instead of first converting behavior into one preference score. Enough clean data and enough model capacity were his concern. The language-model analogy motivates testing scale; it does not show that more compute will recover missing exposure information or solve recommendation.

**Marty's July 13 synthesis proposed a narrower operationalization:** after an observed exposure, predict whether a listener returns within a future window and how intensely they replay the song if they return. This return/count benchmark was Marty's proposal in response to Mo's thesis and raw-behavior thinking, not an adopted decision or an executed experiment. It would use only pre-cutoff behavior and information available then, with count, recency, and metadata kept explicit rather than collapsed into a handcrafted score.

That proposal is different from the existing membership/score hurdle target. It requires an explicit time horizon, exposure definition, censoring policy, and time-correct features. A first observed play reduces exposure ambiguity but may not be the listener's true first encounter. Repeat behavior also remains affected by playlist placement and habit. Success on return prediction would support a narrower claim than discovery of never-before-played music.

Marty's proposed temporal holdouts, strong behavioral baselines, shuffled-audio controls, and model/data scaling ladder would ask whether audio adds predictive information and whether more capacity and evidence improve that result. They are not completed benchmarks or an accepted development roadmap. [Experiments](EXPERIMENTS.md) turns these open questions into small distinguishing tests. The unresolved ambition remains Mo's: learn personally meaningful musical connections without having to specify those connections in advance.
