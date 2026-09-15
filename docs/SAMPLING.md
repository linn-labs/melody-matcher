# What the snowball actually does

[Chronology: why we changed collection](HISTORY.md#6-april-27-and-subsequent-source-revise-the-snowball-then-compare-dataset-theories) · [Dataset experiments](TRAINING_EXPERIMENTS.md#dataset-variants-what-each-change-was-meant-to-test) · [Experiment index](EXPERIMENTS.md)

“Snowball” means expanding a sample from an initial set through related people or items. Here, a target music library supplies artist seeds; their listeners become candidate training users; overlap filters decide which users to collect; qualifying users can supply friends for another pass. It is a way to select training evidence, not the recommender itself.

This explains historical source, not an instruction to collect service data. The [rights boundary](DATA_AND_RIGHTS.md) applies independently of whether the code can access an endpoint. No user data, target library, collection results, or embeddings are supplied.

## Why the original broad collection was not enough

The first collector sought eclectic listeners from a curated artist list and friends. It used tags, genre counts, concentration, and entropy to favor broad histories. After the April listening failure, the working hypothesis was that these histories did not provide enough relevant high-score examples for the target listener.

Adding the target's songs to the embedding catalog addressed a different problem. It made those songs available as representations; it did not ensure that training contained enough examples of a similar context paired with useful high-scored candidates. The snowball aimed to change the latter by finding listeners with stronger overlap, then collecting songs beyond the overlap as well.

That made the collection deliberately target-centered. It could be useful for a personalized research question while also overfitting the entire research process to one listener. It was not a neutral population sample.

## The first plan and its revision

The April 22 proposal described one overlap-filtered expansion, with possible friends/neighbors and repeated rounds. The April 27 revision followed a spot check and replaced that single policy with tagged cohorts. A popularity cap had excluded relevant regions, and weak aggregate overlap could select specialists rather than broadly aligned listeners.

The retained script implements artist-listener discovery and friend expansion. Do not assume every proposed track-listener endpoint, neighbor mechanism, or recursive expansion from the earlier plan exists in the code. The concrete definitions are `STRATEGIES`, seed builders, `discover_candidates`, and `compute_overlap` in [05_taste_snowball.py](../scripts/05_taste_snowball.py).

## The four sampling theories

### Core breadth

Use the target library's leading artists to find candidate listeners. Require track overlap of at least 0.50 **or** artist overlap of at least 0.65. This was intended to favor strongly aligned histories rather than a large number of weakly connected users.

The name describes the intention, not a verified breadth property. The denominator issue below is especially important here.

### Region fill

Choose artists from underrepresented parts of the target taste and use looser thresholds: track overlap at least 0.20 **or** artist overlap at least 0.30. The theory was that a specialist could teach within-region structure even without matching the listener's whole combination of tastes.

Personal-library-derived artist defaults were removed from the curated source. Region-fill seeds must be supplied explicitly by someone with an authorized target and collection method. No private seed list is included in this document.

### Friend expansion

Start from qualifying users in a selected source cohort and examine their friends. Apply the source cohort's thresholds and preserve its identity in the expansion tag. A friend is a candidate to evaluate, not automatically a taste match.

The hypothesis was that social connections might produce more relevant candidates per collection effort than another cold artist-page pass. The code does not demonstrate that this was true. The revised plan described one level; friends-of-friends was a possible later extension, not a proven unlimited snowball.

### Niche deep

Use artists farther down the target library's ranking, with the looser region-oriented overlap thresholds. The revised plan targeted the rank-50–150 region of the artist list and retained a lower listener-count threshold to avoid seeds with little accessible listener material. The theory was that smaller artist audiences might contain highly aligned listeners missed by the leading seeds.

This explored another source of users rather than another neural architecture. Cohort tags were meant to make its eventual value testable separately.

## What “overlap” means in this implementation

`compute_overlap` normalizes the candidate's sampled track titles and artists and compares them with normalized sets from the target library.

- **Track overlap:** matching candidate track entries divided by the number of valid candidate track entries examined.
- **Artist overlap:** distinct candidate artists also present in the target divided by the number of distinct candidate artists examined.
- **Admission:** passing either threshold is sufficient. The two tests are joined by **OR**, not **AND**.

The denominators describe the **candidate sample**. They do not measure how much of the **target library** the candidate covers.

A synthetic counterexample makes the distinction clear: a candidate whose entire sampled history is by one artist already in the target can have full artist overlap while covering only one part of a varied target library. A high overlap threshold therefore cannot guarantee multi-region breadth. The April plan's claim of a mathematical guarantee was wrong under this definition.

This also explains why small or nearly empty histories can look deceptively well aligned. The retained collector includes an activity/library-size gate at collection time. That reduces a particular weak-data case; it does not repair the breadth guarantee.

The filters use normalized names, not perfect recording identities. Variant stripping can merge different versions; missing titles or artist mismatches can change overlap. The target's breadth, candidate sample depth, normalization, and OR rule all affect who passes.

## From a seed to training material

The retained pipeline separates several operations:

1. Build the seed list for the selected strategy.
2. Discover candidate usernames through artist listener pages or friends.
3. Deduplicate and consult progress records.
4. Evaluate sampled top tracks against the target, with per-seed evaluation limits intended to stop the largest seed pools from dominating the sample.
5. For qualifying candidates, collect a fuller set of top tracks with counts and apply collection gates.
6. Store user/cohort/seed/overlap provenance plus user-track records and progress.
7. Match newly introduced tracks and produce embeddings in the separate shared pipeline.
8. Prepare a chosen dataset variant from users and tracks with available representations.

These are distinct populations. Discovered candidates are not all qualifying users; qualifying users do not imply all their tracks have usable embeddings; a tagged user is not necessarily retained by downstream preparation. Published analysis would need those denominators at each stage rather than one headline “dataset size.”

The [component guide](COMPONENTS.md) separately covers operational behavior and execution hazards.

## Why store provenance instead of one final approved cohort?

The revised plan wanted to collect once and test several explanations afterward:

- Does the original pool help, dilute, or dominate the target-relevant signal?
- Is stricter overlap more valuable than a larger moderately aligned pool?
- Do niche listeners contribute useful within-region distinctions?
- Does excluding low-play material improve labels or remove useful weak evidence?

The retained schema stores cohort, seed artist, overlap statistics, and target-library identity. Dataset variants query these fields rather than infer the original collection intent from the current track list. That is the reusable engineering idea here. It does not guarantee perfectly disjoint cohorts or eliminate selection bias.

## What happened and what would resolve it

The April 27 record establishes the revised design and reasons for choosing it. The collector and downstream variants establish implementation. No supplied result ledger establishes the yields of all cohorts, a winning mixture, or a measured recommendation improvement. The sample report in the old plan was an example of desired reporting, not an experiment result.

A useful continuation would freeze evaluation listeners and candidates independently of collection, compare predeclared sampling policies at equal evidence/embedding budgets, and report matching error and coverage separately from predictive quality. Compare on a common target with repeated seeds and per-listener uncertainty. Otherwise a larger overlap score or a different-looking top list can be mistaken for evidence that the model learned taste better.
