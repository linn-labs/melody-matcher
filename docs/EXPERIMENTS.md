# Experiments in the order the questions changed

**[Read the full research history first](HISTORY.md).** This is its compact navigation companion, not a replacement for the explanations. Each stage links to what we believed, what was attempted, what changed our understanding, and what remains unresolved.

**Implemented** means source exists. **Historically reported** means a working record describes an attempt or observation whose reproduction artifacts are absent. **Proposed** means a possible future test, not a completed result. No experiments were rerun for this snapshot.

## 1. Can audio and listening histories provide the evidence?

The early plan paired pretrained audio representations with aggregate Last.fm counts. March collection revealed uneven discovery and an active-listener skew; expired preview URLs then separated “matched” tracks from actually embedded ones.

- **History:** [early plan](HISTORY.md#1-early-plan-learn-taste-from-music-and-listening-behavior) and [March collection](HISTORY.md#2-march-a-broad-dataset-was-already-a-selected-dataset).
- **Detail:** [collection and representations](COMPONENTS.md#collection-and-representations).
- **Status:** collection/embedding source exists; historical checkpoints describe partial progress, not a supplied dataset or a complete sequential model.
- **Unresolved:** what evidence is lost or selected by user discovery, matching, and preview availability?

## 2. Can candidate attention learn what a pooled taste vector misses?

April's design replaced one taste vector with library self-attention and candidate cross-attention. FiLM and score bias made counts influence representation and attention differently. The first supervised task predicted held-out library scores without sampled negatives.

- **History:** [April design and its assumptions](HISTORY.md#3-april-7-9-score-a-candidate-against-the-library).
- **Detail:** [original regression task](TRAINING_EXPERIMENTS.md#the-original-library-regression-task) and [implemented architecture](RESEARCH.md#implemented-architecture).
- **Status:** attention and regression are implemented; the proposed per-listener linear baseline was not completed.
- **Unresolved:** do those mechanisms beat simpler pooling, linear prediction, or less elaborate score conditioning under the same evaluation?

## 3. Does a better validation number mean better recommendations?

The April 22 checkpoint reports an 18-trial sweep with best validation Spearman of 0.719. The subsequent listening test produced poor recommendations clustered in a narrow slow instrumental/background style. The held-out numerical score looked promising; the music sounded bad.

- **History:** [the sweep and listening failure](HISTORY.md#4-april-22-the-number-looked-promising-the-music-was-bad).
- **Detail:** [first sweep](TRAINING_EXPERIMENTS.md#the-first-hyperparameter-sweep) and [actual metric semantics](TRAINING_EXPERIMENTS.md#what-the-metrics-actually-measure).
- **Status:** historically reported result; evaluated weights, logs, recommendation lists, and exact data are absent.
- **Unresolved:** how much of the apparent progress was specific to ranking known-library targets rather than finding satisfying new music?

## 4. Was the model learning the wrong training distribution?

The first response prioritized relevant supervision rather than more tuning. Catalog overlap had been partly engineered and did not prove the training labels covered the listener's taste. A revised snowball collected tagged cohorts based on different sampling theories.

- **History:** [coverage versus supervision](HISTORY.md#5-april-22-separate-catalog-coverage-from-useful-supervision) and [the revised collection](HISTORY.md#6-april-27-and-subsequent-source-revise-the-snowball-then-compare-dataset-theories).
- **Detail:** [what snowball means, step by step](SAMPLING.md), including [the overlap denominator and false breadth guarantee](SAMPLING.md#what-overlap-means-in-this-implementation).
- **Status:** the revised plan and collector exist. Comparative cohort outcomes are not supplied; distribution collapse remains a diagnosis, not a proven cause.
- **Unresolved:** does changing the sample improve prediction on independently fixed listeners and candidates, rather than just change coverage?

## 5. Which users, tracks, and score transforms should supply the targets?

The dataset script defines 17 variants: original versus expanded pools, overlap and diversity filters, activity filters, low-play removal, alternative target transforms, and combinations. A shared recipe was intended to make sampling theories easier to compare.

- **History:** [dataset comparisons after snowball](HISTORY.md#6-april-27-and-subsequent-source-revise-the-snowball-then-compare-dataset-theories).
- **Detail:** [every dataset-variant family and its question](TRAINING_EXPERIMENTS.md#dataset-variants-what-each-change-was-meant-to-test).
- **Status:** implemented definitions, not a verified complete outcome ledger.
- **Unresolved:** which choices help on a common evaluation target? Different transformed targets and selected users make raw metric comparisons misleading.

## 6. What does the model know about candidates outside the library?

The later target plan revisited the no-negatives decision. It introduced random catalog targets and a two-head hurdle variant to separate sampled membership from positive-score regression. Unheard still did not mean disliked; the missing discrimination task had become another cost to measure.

- **History:** [why negatives came back](HISTORY.md#7-later-target-pivot-reintroduce-negatives-and-split-the-heads).
- **Detail:** [negative budgets, losses, ranking choices, and metrics](TRAINING_EXPERIMENTS.md#negative-sampling-and-the-hurdle-model).
- **Status:** five single-head negative budgets and a 270-negative hurdle variant are implemented. Their comparative results are absent; the plan has no reliable experiment date.
- **Unresolved:** can the model discriminate useful candidates without learning false-negative or sampling-prior shortcuts? Does a separate regression head help under a fixed evaluation?

## 7. Are we asking the right question at a meaningful scale?

May's scale discussion and Moritz's July reset moved the focus from product ergonomics to proving the thesis. Use scalable behavioral evidence, try raw counts and metadata rather than a handcrafted preference score, and confront the amount and quality of data needed for a capable network.

- **History:** [the later reset and pause](HISTORY.md#8-may-and-july-reconsider-the-evidence-and-the-scale-of-the-test).
- **Detail:** [Marty's proposed exposure-conditioned return/replay test](FUTURE_PROOF.md), its data contract, baselines, and model/data ladder.
- **Status:** proposal, not an adopted or executed experiment. It is not the target of the existing hurdle code.
- **Unresolved:** can audio improve future behavior prediction beyond popularity and collaborative baselines, and does that improvement translate into music people actually enjoy?

## Before continuing

Read [Components](COMPONENTS.md) for missing artifacts and preserved defects, [Data and rights](DATA_AND_RIGHTS.md) for collection and reuse boundaries, and [Provenance](../PROVENANCE.md) for what this snapshot contains. No application execution, model download, service access, or reproduction success is implied by this reading map.
