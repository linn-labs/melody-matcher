# How the research changed

[Thesis](RESEARCH.md) · [Experiment index](EXPERIMENTS.md) · [Source and artifact map](COMPONENTS.md)

The project did not follow one architecture from start to finish. We kept finding a gap between what we thought the model was learning and what the available evidence actually taught it. This account follows those changes, including the assumptions we abandoned.

The dated checkpoints are historical reports, not reproduced results. The original listening records, trial logs, weights, and private working documents are not distributed. Current source establishes what was implemented, not which experiments completed. Where a plan has no reliable date or the result is missing, that gap stays visible rather than becoming a made-up step in a tidy success story.

## 1. Early plan: learn taste from music and listening behavior

**Starting belief.** A person's collection of loved songs contains connections that genre labels miss. A pretrained audio encoder could give us rich song representations; listening histories could supply evidence about which connections mattered to each person. That would let us learn a preference model without first training an audio encoder or asking thousands of people to rate every song.

The undated initial plan chose Last.fm because it linked tracks and aggregate play counts to individual listeners. Counts were obtainable behavioral evidence, and repeated listening looked like a usable first proxy for preference. It considered MERT, CLAP, OpenL3, and Jukebox, selecting MERT as a music-focused starting point. This was a practical choice, not a completed comparison demonstrating that MERT preserved the information our task needed.

**First proposed architecture.** Feed a set of songs into an attention network, compress the library into one taste vector, then retrieve nearby songs. The plan described this as a two-tower approach and proposed contrastive training with positives and negatives. It also imagined a product around search, previews, and eventual user-library collection.

**What changed.** The collection pipeline became real, but the single-vector retrieval architecture did not remain the design. By April, we wanted candidates to attend to different parts of the same library instead of making one pooled point represent every kind of song someone could love. That was a design rejection, not a failed head-to-head training experiment. [Stage 3](#3-april-7-9-score-a-candidate-against-the-library) explains the replacement.

The early plan used Spotify in its proposed acquisition flow; the March checkpoints describe Deezer. The preserved record establishes that change in provider, but not a sufficiently reliable causal account to claim a particular Spotify restriction forced it. Early statements that short previews were automatically legal to process are not a rights finding. See [Data and rights](DATA_AND_RIGHTS.md).

## 2. March: a broad dataset was already a selected dataset

**What we wanted.** Collect listeners whose tastes crossed categories, obtain their top tracks, and filter for broad listening. The hope was that these libraries would teach the network relationships that single-genre collections would not reveal.

**What we tried.** The March 3 checkpoint describes artist-listener discovery, expansion through friends, aggregate top-track collection, and artist-tag-based diversity scoring. Discovery was not uniform: many artist pages returned errors, and friends lists were not always available. The record reports 5,732 collected users, fewer than originally sought, with a strong skew toward active listeners and deep histories. It records a decision to continue to diversity scoring before immediately repeating failed discovery.

**What we learned.** Having a lot of tracks was not the same as having a representative sample. The way we found people selected for activity; failed pages changed which starting artists contributed users. “Diverse” also meant passing our particular tag, concentration, and entropy rules. None of that established that the resulting supervision represented a new listener's taste.

**The next split in the plan.** By March 5, two methods had distinct purposes:

- **Library method:** use unordered top tracks and counts to learn durable taste.
- **Sequential method:** use timestamped listening events to predict what follows in a listening context.

They could share track matching and audio embeddings, but they were different learning problems. Event collection infrastructure was written; the retained repository does not contain a sequential training model. Do not read the old aspiration to combine them as a completed combined recommender.

**A concrete acquisition failure.** The March 24 checkpoint reports that preview URLs saved during matching had expired before extraction and returned HTTP 403. It also says the embedding file was still empty at that checkpoint. Earlier “pipeline complete” wording therefore meant infrastructure, not a completed embedding corpus. Matching and extraction were consolidated around storing the Deezer track ID and resolving a fresh preview URL near download time.

This matters beyond the operational fix. A matched track is not necessarily an embedded track, and an embedded catalog is not necessarily a good training distribution. Those distinctions became central after the first listening test.

Source: [discovery](../scripts/01_discover_users.py), [diversity scoring](../scripts/03_score_diversity.py), [match and embed](../scripts/04_match_and_embed.py), and [event collection](../sequential/README.md). The [component guide](COMPONENTS.md#collection-and-representations) explains their inputs and hazards without presenting them as approved collection instructions.

## 3. April 7-9: score a candidate against the library

**The objection to the first design.** A library can contain several different kinds of attachment. Averaging HARDY and Aretha into a point between them need not preserve what makes either appealing. A prototype plus dimension weights was also considered, but it still imposed one reference point and one weighting on every candidate.

The April design's requirement was dimensional selectivity: learn which features, or combinations of features, matter for this listener and this candidate. Its examples of individual embedding dimensions were illustrations, not discoveries about what particular MERT coordinates encode. A learned pooled representation was not experimentally disproved; it remains a useful baseline.

**What we built instead.** Keep a representation for each context song. Let the library songs inform one another through self-attention, then let each candidate cross-attend to that encoded library. Different candidates can draw on different context songs without reducing the entire library to one retrieval point.

Play-count-derived scores entered in two ways:

1. **FiLM:** each score produces feature-wise scales and shifts for that song's embedding. The motivation was to let a score change how features are expressed, rather than append it as another audio coordinate or multiply every coordinate by the same amount.
2. **Attention bias:** a context song's score is added to its key attention logits, giving high-scored songs additional influence inside library attention.

Those are distinct mechanisms. Their presence is verified in source; neither has a completed ablation here showing that its complexity improves recommendations. [Research](RESEARCH.md#implemented-architecture) gives the dimensions and exact structure; [model.py](../library/src/model.py) and [layers.py](../library/src/layers.py) are the definitions.

**The first training task.** Reveal part of a listener's library and predict transformed play counts for the held-out songs. Log normalization compressed the difference between casual plays and obsessive replay while retaining order. Randomly changing the context/target partition produced more training examples from the same libraries, not more independent listeners or new preference evidence.

**The assumption that later broke.** We rejected random negatives because a song absent from a library is not necessarily disliked. It may simply be undiscovered. The April design then argued that relative score prediction among known songs should be enough: low-play tracks would provide weaker positive evidence, and the learned ordering would extend to the catalog.

That second step was a hope, not a consequence of the first. Knowing that unplayed does not mean disliked did not establish that training only on played songs would teach catalog-wide discrimination. The April document explicitly noticed that the model never saw true zero targets and initially treated relative ranking as sufficient. [Stage 7](#7-later-target-pivot-reintroduce-negatives-and-split-the-heads) returns to that decision.

**An alternative left unfinished.** An April 9 note proposed fitting linear weights to each listener's embeddings and play-count targets. It would test whether simple feature weighting carried useful signal before crediting attention for it. The baseline was designed but not implemented. Failure of a linear fit would not by itself prove that a neural network was necessary; bad labels, insufficient data, or weak representations could also make it fail.

## 4. April 22: the number looked promising; the music was bad

**What we were trying to improve.** The model was trained to predict held-out scores. A hyperparameter sweep varied optimization settings and architecture choices to improve that task.

**What the checkpoint reports.** An 18-trial sweep reached a best validation Spearman of **0.719** in a longer-training trial. Spearman measures rank agreement; it is not a percentage of recommendations that someone liked. The checkpoint described a narrow range among the stronger trials and interpreted it as a plateau for that recipe and dataset. That was not a controlled scaling study.

**What we did next.** Build a harness around a real imported library, use the trained scorer across the catalog, inspect the top recommendations, and listen. The harness could switch checkpoints and record feedback rather than making us infer usefulness from a validation number.

The reported recommendations clustered in slow instrumental/background music and did not match the listener's taste. They were unfamiliar, but unfamiliarity was not the failure—we wanted discovery. The failure was listening to them and finding that they did not hit.

**What this changed.** We could no longer treat predicting held-out library scores as a sufficient proxy for finding satisfying new music. The numerical evaluation had looked promising while the recommendations sounded bad. That is the entire meaning of the earlier shorthand “offline/listening disagreement.”

The report prompted a shift from tuning the network to examining the training sample and supervised task. It did not identify a unique root cause. The evaluated weights, exact splits, trial logs, and recommendation lists are absent, and later source changes mean this snapshot is not a replay of that experiment.

The retained [harness](../harness/README.md) is the inspection source, not a hosted demo. Its historical “Play” control opens an Apple Music search link; it does not implement in-app playback. The [metric definitions](RESEARCH.md#implemented-targets-and-evaluation) explain the current code's aggregation and target limitations.

## 5. April 22: separate catalog coverage from useful supervision

**The working diagnosis.** The retraining plan called the failure “training-distribution collapse”: perhaps the high-score training signal favored a narrow audio region, so the model learned to favor it even with an unsuitable input library. That diagnosis drove the next work. It was stated far more confidently in the original notes than the available evidence warrants; it remains a hypothesis rather than a demonstrated causal result.

The accompanying discussion corrected three arguments that had obscured the problem:

1. **Finding the listener's songs in the catalog did not prove the sample covered their taste.** An earlier targeted collection pass had deliberately improved overlap. High matching coverage was partly an engineered outcome. It did not tell us whether the rest of the catalog contained useful discoveries or whether training taught the right conditional scores.
2. **A rich audio representation did not guarantee the downstream model would use it well.** The same song vector can be available everywhere while the learned mapping from library and candidate to score is poorly supported by relevant examples.
3. **Changing the inference catalog did not repair the training function.** Restricting candidates could alter what was returned, and might help a retrieval problem, but it would not itself teach a missing preference relationship. The original claim that filtering could never help was too absolute; the useful distinction is between changing available candidates and learning better scores.

**The choice.** The plan considered inference filtering, reweighting or fine-tuning existing examples, and collecting new users with stronger target-library overlap. It favored new collection, while proposing reweighting as a cheaper diagnostic during embedding extraction: was useful supervision already present but diluted, or did we need more of it?

**Why “more users” was not enough.** If the existing collection overrepresented irrelevant behavior, adding more of the same could reinforce the mistake. The next collection was intended to change the composition of supervision, not merely enlarge a catalog.

This led to taste-biased snowball sampling. The original four-phase plan was collection → embeddings → intermediate existing-data diagnostic → retraining, with the diagnostic intended to overlap extraction. That was a plan. The surviving code implements collection and dataset comparisons, but the archive does not establish that every proposed phase or fine-tuning variant ran to completion.

## 6. April 27 and subsequent source: revise the snowball, then compare dataset theories

**First snowball idea.** Start from artists or tracks in the target library, find listeners, retain those with enough overlap, collect their fuller histories, then expand through qualifying listeners' friends. The purpose was to find people whose *other* tracks might supply the missing supervision. [The snowball deep dive](SAMPLING.md) explains the actual retained mechanism, thresholds, and source paths.

**What the spot check changed.** The April 27 plan records two problems with the first design. A popularity ceiling intended to avoid noisy artist pages excluded whole relevant regions of the target taste. A modest aggregate overlap threshold could admit a specialist sharing only one slice of a broad library. “Has some of my songs” was not the same as “has the combination of tastes we want the model to learn.”

**The revised plan.** Collect several tagged cohorts rather than betting everything on one threshold:

- **Core breadth:** stricter overlap, intended to find strongly aligned libraries.
- **Region fill:** looser overlap around underrepresented areas, admitting specialists on purpose.
- **Friend expansion:** search outward from qualifying users and preserve the originating filter.
- **Niche deep:** seed from farther down the target library's artist list to find listeners the largest favorites missed.

Tagging users and storing overlap statistics would let later training experiments select different combinations without repeating the entire collection. The accepted tradeoff was more upfront collection and embedding work in return for room to test several theories downstream.

**A mistake worth retaining explicitly.** The plan called its strict overlap threshold a mathematical guarantee of breadth. The implementation does not provide that guarantee. Its denominator is the candidate's sampled tracks or artists, not the target library, and admission is track overlap **or** artist overlap. A specialist can therefore overlap strongly with one slice of the target and still pass. Raising the threshold strengthens alignment under that definition; it does not prove coverage across the target's taste regions. This is a retrospective reading of the code, not a claim that we measured the effect then.

**The implemented dataset experiments.** The next source makes collection feed a set of comparisons: keep the original cohort, use snowball users, tighten overlap, favor diverse or engaged listeners, remove low-play tracks, change score transforms, or stack several of those choices. The script defines 17 variants, with an ordering intended to produce informative comparisons even if a run stopped early. [Training experiments](TRAINING_EXPERIMENTS.md#dataset-variants-what-each-change-was-meant-to-test) explains every family and variant.

The script's recipe is based on the earlier longer-training trial, with a larger batch and shorter early-stopping patience. Holding that recipe fixed was intended to isolate dataset differences. It does not make the resulting metrics directly causal or comparable: variants change users, targets, sample size, and task difficulty.

**What we can conclude.** The multi-cohort collector and comparison machinery exist. A complete result ledger establishing which cohort or transform won is not supplied. We cannot fill that gap by treating illustrative reports in a plan as measurements or by assuming a script's presence proves its full sweep completed.

## 7. Later target pivot: reintroduce negatives and split the heads

**The new concern.** A later plan, without a reliable experiment date in its body, focuses on a different mismatch. During training, every candidate had been a held-out member of the same user's library. At inference, candidates came from the whole catalog. Even a well-chosen listener cohort did not make those candidate distributions the same.

The note describes a score plateau and duplicate clusters during inference inspection, motivating catalog-negative experiments. It is not an archived scored recommendation list or a controlled comparison showing that the snowball caused, or failed to fix, those symptoms. The defensible sequence is that this plan revisited the earlier no-negatives choice and built on the dataset-experiment machinery—not that every sampling experiment finished before this thought occurred.

**First remedy: introduce a rejection task.** Append random catalog tracks with zero targets to the held-out positives. Sweep negative counts of 0, 30, 90, 270, and 900 while keeping the selected dataset and model backbone fixed. The question was whether exposing the model to out-of-library candidates would improve discrimination, and how sensitive it was to the negative budget.

**The next problem, anticipated in the design.** With many zero targets, a single mean-squared-error objective can make predicting low scores broadly attractive. Lower loss would not necessarily mean better discovery. That motivated a second variant rather than simply turning up the number of negatives.

**Second remedy: a hurdle model.** Keep the attention backbone but split the outputs:

- a classifier for positive prepared-library targets versus sampled negatives;
- a regressor trained only on the transformed scores of positive targets.

The historical language was “would they play it?” and “given that, how much?” The actual labels are narrower: sampled membership and normalized aggregate counts. They do not measure known exposure, calibrated real-world play probability, or future replay intensity.

The harness can rank the same hurdle checkpoint by classifier probability, probability times score, or score above a probability threshold. That separates the effect of the learned heads from the choice of ranking rule. [The detailed account](TRAINING_EXPERIMENTS.md#negative-sampling-and-the-hurdle-model) explains the losses, controls, metrics, and pitfalls.

**What changed in our reasoning.** April's concern had not become false. Unheard still did not mean disliked. We had instead recognized the cost of the alternative: positive-only score regression left catalog discrimination weakly specified. Implementing random negatives made this a tradeoff to test, not a resolved philosophical objection.

**What remains unknown.** The variants, heads, and ranking choices are implemented. Their comparative results are not supplied. The July 13 archive index itself says execution and results still need verification. No winner or successful repair can be reconstructed from that evidence.

## 8. May and July: reconsider the evidence and the scale of the test

The May discussion raised a broader doubt: how capable would a language model be with the equivalent of this model size and amount of data? Poor performance at this scale might say less about the originating thesis than we were assuming. It also distinguished arbitrary order in a library from meaningful order in listening events. These thoughts sharpened the interpretation; they did not originate April's already implemented set-like architecture.

**Moritz's July 13 reset.** Stop designing the final recommendation product and find the most straightforward way to prove the idea. Whether someone imports a library or selects songs can wait. The evidence must scale, so asking everyone to score songs from one to ten is not a satisfactory foundation.

Instead of turning plays into an invented preference score first, try the embeddings and actual behavior, then add metadata such as time since addition, song length, and release age. Let the model learn which information matters. The larger-model ambition came with its own constraint: where would enough correct training data come from?

This was not a report that a raw-count model had been trained. Nor did the language-model analogy establish a music scaling law. It explained why the current failure did not settle the larger question and why data access had become the main obstacle.

**Marty's proposed first test.** Predict return and future replay after a known observed encounter, using only information available before a cutoff. This would reduce the ambiguity of treating a never-heard song as a rejection and retain count, recency, and metadata as separate inputs. It is an unadopted proposal, not the target implemented by the existing hurdle model. [Future proof](FUTURE_PROOF.md) sets out its data contract, baselines, and remaining limitations.

**Where that leaves the project.** We have a thesis worth pursuing and several useful attempts, not a working recommender. Our current diagnosis is that the available preference evidence may be inadequate or wrong for the task, and the downstream model may already be too large for it. The broader ambition may need much greater scale. Increasing parameters without increasing the quality and amount of evidence would not answer that problem.

The most useful continuation is not to pretend the last architecture was one tune away from success. It is to choose a defensible prediction target and obtain the evidence needed to distinguish sampling, representation, objective, and capacity explanations. The [experiment index](EXPERIMENTS.md) connects each historical stage to that unresolved question.
