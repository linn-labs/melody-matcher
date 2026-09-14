# Experiments and open questions

**Implemented** below means source exists, not that this release ran it. **Historically reported** means a dated record describes an attempt or observation with unavailable reproduction artifacts. **Proposed** means a future test. No new results were produced for this snapshot.

## Sampling and coverage

**Implemented:** [discovery](../scripts/01_discover_users.py), [diversity scoring](../scripts/03_score_diversity.py), [taste snowball](../scripts/05_taste_snowball.py), [spot checks](../scripts/spot_check_snowball.py), and [dataset variants](../library/scripts/dataset_experiments.py). Source includes broad curated seeds, diversity gates, target-library overlap cohorts, friend expansion, and long-tail seed selection. Personal-library-derived region-fill defaults were removed; that strategy now needs explicitly supplied seeds.

**Motivation and evidence:** the April 22, 2026 listening report described poor, clustered recommendations and suspected sampling bias. The collection and variant code shows attempts to change coverage, but no retained dataset or trial artifacts establish their effect.

**Confounds:** seed popularity, preview availability, fuzzy-match errors, user activity, overlapping cohorts, and selection around a single target library can all change the problem. The SQL filter admits users with missing diversity scores regardless of cohort, despite its narrower comment.

**Smallest proposed discriminator:** compare two predeclared sampling policies at equal event and embedding budgets, on the same frozen evaluation listeners/candidates. Report coverage and matching error separately from predictive lift, with per-listener uncertainty. A broader sample improving both coverage and controlled outcomes would support a sampling explanation more strongly than a new top-50 anecdote.

## Target transforms, negatives, and hurdle heads

**Implemented:** four score transforms and track filters in [preprocessing](../library/src/preprocessing.py); 17 dataset definitions in [dataset experiments](../library/scripts/dataset_experiments.py); negative counts 0, 30, 90, 270, and 900 plus a 270-negative hurdle variant in [negative experiments](../library/scripts/negative_sampling_experiments.py). The harness supports probability, joint-product, and thresholded-score ranking.

**Motivation and evidence:** within-library score regression may not separate desirable catalog songs from merely absent ones. Source establishes these experimental mechanisms; there are no distributed results showing that one solved the reported failure. Historical narratives offered conflicting negative-sampling explanations; neither is accepted as settled.

**Confounds:** absence does not imply dislike. Prepared-library filtering can remove known tracks before negative rejection. Negative ratios alter the classification prior; duplicate draws and random evaluation windows affect metrics. Different score transforms define different targets, so comparing their raw losses or correlations is not an apples-to-apples preference test. Some ratio labels in the source reverse positive/negative notation; the configured counts are the reliable specification.

**Smallest proposed discriminator:** freeze one common evaluation target/candidate construction, then compare regression with and without negatives and the two-head loss across repeated seeds. Evaluate ranking and calibration under the evaluation sampling prior. Keep a separate audit of known-track false negatives; do not interpret the classifier as exposure-aware.

## Exposure-conditioned return and replay

**Proposed by Marty, July 13, 2026, in response to Mo's thesis and raw-behavior proposal; not an adopted or executed experiment:** predict return within a fixed future horizon and conditional future play count/rate after a known observed exposure. No retained trainer or prepared dataset implements this target. The event collector and schema are only potential inputs.

**Motivation:** keep behavioral evidence closer to its recorded form and reduce the ambiguity between an unseen song and a rejected song. Counts still reflect opportunity, activity, habit, and availability.

**Smallest proposed discriminator:** construct a consented or otherwise appropriately authorized pilot with one frozen cutoff/horizon policy. Compare a popularity/activity/prior-count baseline to an audio-aware model using only pre-cutoff features. Define observation coverage, censoring, missing events, and what “first exposure” means before fitting anything. Measure return calibration and conditional-count error separately; a Poisson count baseline and an over-dispersed alternative are proposals, not existing heads.

## Set versus sequence and dimensional selectivity

**Implemented:** score-conditioned FiLM, self-attention without positional encoding, and candidate cross-attention in [model.py](../library/src/model.py). [Scrobble collection](../sequential/scripts/01_collect_scrobbles.py) records timestamps but does not supply a sequence model.

**Motivation:** aggregate taste need not depend on arbitrary library order; immediate listening context may depend on actual event order. Candidate attention could select relevant parts of a diverse history, but that interpretation has not been measured.

**Smallest proposed discriminator:** first test permutation consistency with dropout disabled and embeddings/scores/masks permuted together. Then compare a pooled-history baseline, the current attention structure, and ablations of FiLM and score bias at matched budgets. A separately designed timestamp-aware sequence branch should be compared on session prediction, with an explicit distinction from durable-taste evaluation. Inspect predictive changes under ablation rather than treating attention maps as explanations.

## Model and data scale

This is the main reason the project is paused: we suspect the downstream model is too large for the user evidence we have, while the larger version of the thesis needs far more and better data than we can currently obtain. Shrinking the model is a useful diagnostic; scaling parameters without scaling the evidence is not the experiment we want. This is our current hypothesis, not a completed model-size comparison. [The research account](RESEARCH.md) explains how we got here.

**Implemented:** [hyperparameter sweep definitions](../library/scripts/hyperparam_sweep.py) vary optimization and architecture, including hidden width and depth. **Historically reported:** the April checkpoint reported an 18-trial sweep and a 0.719 best validation Spearman. It supplies no controlled scaling law, and the current sweep has a preserved return-value mismatch described in [Components](COMPONENTS.md).

**Confounds:** data quality, label definition, training compute, optimization, and evaluation difficulty can dominate apparent size effects. The selected external MERT encoder's 330M designation is not the size of the downstream taste model.

**Smallest proposed discriminator:** freeze a leakage-safe task and fit the same architecture family at several model/data budgets with repeated seeds. Plot held-out improvement against event count, parameter count, and compute. Scale only where the target outcome improves; training-loss improvement alone would not support the thesis.

## Evaluation and listening

**Implemented:** user-ID splits, random context/target construction, score metrics, and a qualitative comparison harness. **Historically reported:** offline/listening disagreement on April 22, 2026. **Proposed:** out-of-time evaluation, cold-song and held-out-listener slices, and a blinded listening protocol.

The smallest credible audio-value test compares popularity/activity, collaborative behavior, pooled audio similarity, a linear predictor, metadata-only neural prediction, audio plus behavior, and audio plus behavior plus metadata under one task definition. A shuffled-audio control should preserve other features while breaking track/audio correspondence. These baselines are proposed, not supplied as completed code or measurements.

Predeclare candidate construction, horizon, metric aggregation, calibration, and less-popular/cross-genre slices. Shuffling or fitting preprocessing across train/test boundaries can itself leak information. Pair offline outcomes with consented, blinded listening judgments; familiarity and satisfaction should be recorded separately. Audio-specific lift over strong baselines would support a narrower predictive claim, not automatically prove a useful discovery product.
