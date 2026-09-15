# What the training experiments were testing

[Chronological account](HISTORY.md) · [Sampling detail](SAMPLING.md) · [Experiment index](EXPERIMENTS.md) · [Future proof](FUTURE_PROOF.md)

This is the technical companion to the learning history. The sequence was: predict scores within a library, tune that model, inspect its catalog recommendations, change the training sample and target transforms, then revisit catalog negatives and split the output heads. Those later mechanisms exist in source, but their comparative results are not supplied. A configured experiment is not a completed experiment.

## The original library-regression task

For each user, preparation selects tracks with available embeddings and transforms aggregate play counts into scalar scores. The default is `log(1 + count)` divided by the maximum of that quantity within the prepared library. It compresses count extremes without reversing order. It does not separate affection from age, exposure, background listening, or activity.

[TasteDataset](../library/src/dataset.py) takes a random window of up to 300 tracks by default, shuffles it, and uses 70% as context and the remainder as targets. Context contains both embeddings and scores. The network predicts scores for the held-out embeddings. A new random split lets the same listener provide different context/target combinations across steps.

The [model](../library/src/model.py) conditions each context embedding with FiLM, applies score-biased self-attention across the library, and cross-attends each candidate to the encoded context. A scalar regression head predicts its score. The [architecture section](RESEARCH.md#implemented-architecture) gives the actual dimensions and layer counts.

Users are split by numeric `user_id % 10` into training, validation, and test groups. That can separate users, but is not a future-behavior test. The target transform is calculated before the context/target split, so normalization uses the full prepared library, including held-out songs. Different random windows are not independent new preference observations.

**Why we tried it:** a tractable supervised task from available counts, without inventing dislikes for songs the listener had never heard.

**What it did not ask:** whether a random catalog candidate was relevant, whether someone would replay a new song in the future, or whether listening to the recommendations would be satisfying.

## The first hyperparameter sweep

[hyperparam_sweep.py](../library/scripts/hyperparam_sweep.py) contains optimization and architectural comparisons. The April 22 checkpoint reports an 18-trial sweep and a best validation Spearman of 0.719 for a longer-training trial. The stronger trials occupied a narrow reported range, which motivated looking beyond minor recipe changes.

That result belongs to the historical checkpoint, not to a run of this curated snapshot. The original experiment was not rerun for this release.

The listening test then found poor, narrowly clustered recommendations. A decent rank correlation among held-out library targets had not established useful catalog ranking. That observation motivated the dataset comparisons below; it did not prove that hyperparameters, representation, or model capacity were irrelevant.

## Dataset variants: what each change was meant to test

The [dataset-experiment script](../library/scripts/dataset_experiments.py) defines 17 variants. Its shared training recipe was intended to make the data-selection and target choices the main comparisons rather than retuning the network for each one.

### Change which users supply supervision

- **`baseline_all_diverse`:** the default diversity-filtered pool, without a cohort restriction. This is the common reference, not proof that every retained user is broadly eclectic; the SQL also admits missing diversity scores.
- **`original_17k_only`:** select the `original` cohort. Intended to compare the pre-snowball pool with the expanded sample. The historical name is not proof of an exact user count or reconstruction of the old trial's bytes.
- **`high_overlap_broad`:** track overlap at least 0.20, artist overlap at least 0.30, or the `niche-deep` cohort. Tests more moderately aligned supervision, with that explicit niche exception.
- **`high_overlap_strict`:** select `core-breadth`. Tests stronger alignment under the stored cohort definition, not a verified guarantee of taste breadth.
- **`niche_deep_only`:** select `niche-deep`. Tests whether the deeper seed strategy contributes useful distinctions.
- **`diverse_listeners_strict`:** raise the genre requirement to five, require entropy at least 6.0, and lower top-five concentration below 0.40. Tests stronger eclecticism filtering, subject to the actual preprocessing SQL and missing-value behavior.
- **`engaged_users`:** require at least 10,000 total scrobbles. Tests richer activity as a source of evidence; activity is not automatically cleaner preference.
- **`non_original_snowball_only`:** exclude the original cohort. Tests whether the target-centered collection works better without the initial pool.

[Sampling](SAMPLING.md#what-overlap-means-in-this-implementation) explains why high overlap can still admit a specialist and why these are different selection theories, not a ranking of data quality.

### Change which tracks count as evidence

- **`drop_low_play_tracks`:** remove tracks with fewer than three plays. Tests whether weak-count tails are mostly noise.
- **`drop_one_play_tracks`:** remove single-play tracks. A milder version of the same idea.
- **`top_100_per_user`:** retain only the leading 100 tracks by count. Tests a concentrated preference signal at the cost of breadth and weak evidence.

The tradeoff is not simply clean versus dirty data. A new favorite can have few plays, while an old background track can have many. Filtering also changes the library that the negative sampler later treats as known.

### Change what score the model must predict

- **`rank_percentile_scores`:** replace log-normalized magnitudes with within-user ranks. This reduces the effect of count extremes but discards magnitude differences. The retained implementation gives distinct ranks to ties.
- **`binary_top_quartile`:** predict a top-quarter indicator. This asks for a coarse distinction rather than precise relative intensity.
- **`z_score_per_user`:** standardize log counts, clip, and rescale to the unit interval. This changes how unusual a track's count must be within that listener to receive a high target.

These transformations alter the target itself. Lower MSE on a binary target and lower MSE on a continuous target are not directly comparable measures of preference learning. Nor does a better correlation on an easier transformed task establish better discovery.

### Combine the changes

- **`me_overlap_clean`:** the broad-overlap selection plus removal of tracks with fewer than three plays.
- **`me_centric_maximal`:** select core/niche cohorts or artist overlap at least 0.30, require a larger minimum library, and remove tracks with fewer than three plays.
- **`diverse_engaged_clean`:** combine stronger diversity, activity, and low-play filtering.

Combined variants test practical packages of choices. They cannot alone attribute a change in outcome to one ingredient. The names retain historical intent, not established findings about a target listener.

### What the comparison can and cannot tell us

Keeping model settings fixed controls one part of the experiment. It does not freeze listener identities, event volume, target scale, candidate distribution, or evaluation difficulty. The [component guide](COMPONENTS.md#preserved-defects-and-hazards) separately records implementation limitations for anyone inspecting or continuing the source.

The source establishes the variants and their intended questions. The archive does not supply a verified complete outcome for each one. A continuation should evaluate them on a common independently frozen task, report retained users/tracks and preprocessing identities, and distinguish improved coverage from improved prediction.

## Negative sampling and the hurdle model

The later [negative-experiment script](../library/scripts/negative_sampling_experiments.py) asks a different question: does training only on library members leave the scorer unable to distinguish relevant songs from other catalog candidates?

### Single-head comparisons

Keep the selected `baseline_all_diverse` dataset and attention backbone. Append `K` catalog candidates with score zero to the held-out positive targets, for `K` equal to 0, 30, 90, 270, or 900. Every nonpadding target contributes to ordinary MSE. With the default full window, there are 90 held-out positive targets, but shorter libraries can produce fewer; the configured counts are more reliable than informal ratio labels in comments.

The negative sampler rejects members of the **prepared** library, not every song ever observed for that user. A known track removed during filtering may become eligible as a negative. Repeated draws are allowed. These labels mean sampled absence from the prepared set, not observed dislike.

The intended comparison was a sensitivity test: perhaps some negative evidence would improve catalog discrimination, while too many zeros would push the regressor toward generally low predictions. That latter behavior was a risk the plan wanted to measure, not a reported outcome for every high-negative trial.

### Why two heads?

The hurdle variant uses 270 sampled negatives and the same basic encoder/candidate backbone, but separates two objectives:

- **Binary cross-entropy:** classify prepared-library positives versus sampled negatives using `play_logits`.
- **Positive-only MSE:** predict transformed counts using `score_pred`, without making the regression head fit the many zero-negative targets.

The total loss adds the classification loss and weighted positive-score loss; the initial `hurdle_alpha` is 1.0. This removes negative zeros from the regression loss. It does not guarantee that the shared representation, classifier, or combined ranking cannot collapse or learn sampling shortcuts.

The historical label `P(played)` is `sigmoid(play_logits)`. It is not a calibrated probability that someone will play a never-heard song. The classification prior depends on the sampled negative budget, and the regressor predicts a transformed existing-library score, not a future count.

### Three rankings from the same checkpoint

[harness/backend/inference.py](../harness/backend/inference.py) supports:

- **Probability:** rank by the classifier output alone.
- **Joint:** rank by classifier probability multiplied by the score-head prediction.
- **Threshold:** reject candidates below the probability threshold, then rank the remainder by the score head.

This makes it possible to ask whether the two heads contain different useful signals without retraining for every presentation. It does not make their product a validated expected-play count. The regression target is transformed, and calibration is unresolved.

### What the metrics actually measure

The authoritative definitions are in [train.py](../library/scripts/train.py):

- **Spearman:** pooled rank correlation on nonpadding positive targets. Training concatenates up to the last ten batches; validation/test concatenate all loader batches. It is not mean per-listener correlation. For hurdle models the ranking input is the classifier probability, not the score head.
- **AUC:** discrimination between positive and sampled-negative targets under that sampling construction and pooled aggregation. Easy random negatives can make this look good without proving useful full-catalog discovery.
- **NDCG@50:** ranking quality on constructed target sets. The candidate construction matters; it is not a catalog-wide temporal recommendation benchmark.

Validation windows and negatives are resampled. Different negative counts, target transforms, or listener pools change both the learning problem and the metric's difficulty. A fixed dataset/backbone is a useful experimental intention, but a serious comparison still needs fixed evaluation cases and repeated seeds.

The heads, negative budgets, and harness rankings are implemented. The original plan's example metrics and expected top-list behaviors were specifications and predictions, not measured results. No winning budget, ranking, or successful repair is established here.

## Alternatives that remain open

The April linear baseline was proposed, not completed. A regularized per-listener predictor would help test whether attention adds value over a simple mapping, without assuming a failed linear fit proves the thesis wrong or the network necessary.

Other useful comparisons include pooled-library similarity, FiLM versus concatenation or uniform score weighting, attention-bias ablation, and permutation consistency with dropout disabled and embeddings/scores/masks permuted together. Attention maps alone would not establish an explanation of taste.

The July proposal changes the task more fundamentally: predict future return and replay after observed exposure, rather than membership and normalized current counts. That requires different data and evaluation. [Future proof](FUTURE_PROOF.md) describes it separately so it cannot be mistaken for the implemented hurdle experiment.
