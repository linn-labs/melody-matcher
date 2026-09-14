# The thesis, the attempts, and why we stopped

## We haven't run out of good music

Spotify and Apple Music recommendations suck. I keep finding songs I'm absolutely in love with that were released years, sometimes decades, ago. They existed the whole time. They just never got recommended to me.

So no, the problem isn't that there's only so much good music out there. Every old song that becomes a new obsession is something the recommendation system missed. There's a lot left to find.

Nearest-neighbor search is part of how these recommendations are made. Spotify [describes using it to power recommendations](https://engineering.atspotify.com/2023/10/introducing-voyager-spotifys-new-nearest-neighbor-search-library), including Discover Weekly and Home. And "danceability" isn't a made-up example: Spotify [defines it as a numerical audio feature](https://developer.spotify.com/documentation/web-api/reference/get-audio-features). Its now-deprecated [Recommendations API](https://developer.spotify.com/documentation/web-api/reference/get-recommendations) exposed a target danceability and explicitly preferred tracks whose attributes were closer to those targets.

But being similarly danceable is a shitty explanation for why two songs belong together for someone. So are genre labels on their own. Nearest-neighbor search finds what's close in the representation it's given. The hard part is learning a representation that captures why a particular person loves a particular song. Faster search doesn't fix a bad definition of similarity.

## Music DNA

I love both HARDY's "Favorite Country Song" and Aretha Franklin's "Don't Play That Song." I can't give you a clean explanation for why. Country and soul don't get me very far, and putting those labels next to each other doesn't explain the connection either.

Taste isn't random. There's something in the music's DNA that explains why both hit. It might be in a voice, a bit of tension, the way a phrase lands, or a combination of things we don't have words for. Let the model learn those connections rather than deciding in advance which descriptors explain them.

Think about recommending music to a friend you know well. Sometimes you hear a song and immediately know: *they're going to love this*. You aren't calculating its danceability. You've built a representation of that person's taste from the music you've heard them love, and you have a representation of the new song even if you're hopeless at explaining either one. Our brains already make connections we struggle to describe. That's what the artificial neural network should learn to do.

Music DNA is the underlying information that makes those connections possible. Audio embeddings are our starting representation. The question is whether they preserve enough of what matters to a listener.

Train a sufficiently capable network on rich representations of the music and enough real listening behavior, and let it learn why apparently unrelated songs belong together for a particular person. We shouldn't have to teach it genre rules or explain in words what makes a song hit.

That's the thesis. We haven't made it work yet.

## More of this feeling, without playing the song to death

There's more to this than recommending another random song you might enjoy.

Sometimes you're completely obsessed with one song. You want to keep getting whatever it gives you, but you don't want to listen to it so much that you ruin it for yourself. *Find something else that scratches this particular itch.*

That calls for a song closer in the parts of musical DNA that matter right now, not just another track with high danceability. A useful learned space would let you stay close to that obsession or reach further out while still finding something you'll love.

Closeness becomes useful when the representation captures the relationship between the music and the listener. Learning that relationship is the job; a handful of descriptors can't do it for us.

The April design proposed emphasizing one context song's score to ask for "more like this." That control remains an idea; the retained code scores candidates against a library.

## Music is the first test

If this works for music, the same principle can extend to nearly anything we'd want recommended to us. Books, movies, maybe even clothes. Represent the thing as a meaningful vector, then train a large neural network on people's preferences against those representations. Let it learn the connections instead of asking us to explain our taste first.

Two books can leave you with the same feeling without sharing a genre or a plot. Two movies can have almost nothing in common in a catalog and still be exactly your kind of movie. With clothes, you can look at something and immediately know it's you, even if you'd never have thought to search for it. The descriptions we use to sort things aren't the whole reason we love them.

The inputs change with the medium. A book gives us text; a film gives us images, dialogue, sound, and how they unfold together. Clothes bring in how something looks, feels, and fits. A purchase isn't automatically love, any more than a play is. Each model needs the information that matters to that relationship.

Music is the first test of a much broader ambition: rich representations of things, real evidence of people's preferences, and a network capable of learning the connections between them.

## Why it's paused

We paused because we couldn't see a convincing route to the data needed to test the idea properly.

The latest hypothesis is mainly a data problem: we don't have the right user listening data, or enough of it, and the downstream network may already be too big for the evidence we can feed it. A model can be too big for the available dataset and still be nowhere near the scale needed for the larger ambition.

Scaling the NN requires scaling the data. We don't currently have a way to get the quality and volume we'd need. Access to listening behavior and usable song recordings is the constraint. A smaller model would be a useful comparison, but wouldn't supply the missing evidence.

The problem kept turning up in different forms. Who did we collect? Which of their songs could we represent? Did lots of plays mean love, years of playlist placement, or simply more opportunities to hear the track? Was an absent song unwanted or just undiscovered? Was the model learning taste, or learning the shape of our sample? Changing the loss doesn't settle those questions, and collecting more of the same biased signal can reinforce the problem.

The earlier records focused on sampling. Later conversations broadened the concern to the data/model balance and what we were training the model to predict. The controlled comparisons needed to separate those explanations haven't been run.

By July 13, the question had shifted from product design to proving the idea as quickly as possible. Forget whether someone imports a whole library or selects a few songs. The inputs have to scale; asking everyone to rate every song from one to ten won't do. Try audio embeddings and actual plays, add useful metadata, and let the model learn what matters instead of handing it an invented preference score.

We're sharing the attempts and mistakes so someone with the data, access, or a better idea can take this further.

## The engineering trail

The account below follows the design decisions, historical reports, and remaining experiments. Original data, trained weights, and trial logs aren't included; the experiments weren't rerun for this release.

## How the approach took shape

### Early plan and March: obtain behavioral evidence and usable audio

We started with a fairly practical plan: pair a pretrained audio representation with Last.fm track counts, then train a two-tower model that summarizes a listener into one vector for retrieval. Counts were something we could actually collect without asking people to sit down and score their entire music library. They gave us a starting signal. The question was how much of taste that signal really captured.

We considered MERT, CLAP, OpenL3, and Jukebox, and chose MERT as the starting representation because it was music-focused and manageable to work with. Training our own audio encoder first would have made an already difficult project much harder. The extractor still uses MERT; its actual dimensions are listed below, correcting some early notes.

The **March 3, 2026 checkpoint** described discovery aimed at listeners with broad tastes, followed by aggregate top-track collection and diversity filtering. The intention was to expose the model to connections across musical categories. The same record reported uneven discovery yield and a strong skew toward active listeners with deep histories. It recorded a decision to proceed with the collected pool rather than immediately repeat failed discovery attempts, pending the diversity results. Coverage was already a research constraint, not just a count of collected users.

The **March 5 checkpoint** separated two questions: durable taste from unordered libraries, and next-song context from chronological events. It described Deezer preview matching and timestamped collection infrastructure. The **March 24 checkpoint** then reported that saved preview URLs had expired before embedding extraction; matching and extraction were consolidated so previews could be resolved near use. Resolving previews near use fixed that acquisition problem. Current collection requirements are covered in [Data and rights](DATA_AND_RIGHTS.md).

### April design: score each candidate against the library

By the **April 7 principles and April 9 scoring design**, we had rejected the single-vector starting point. A crude average of Aretha and HARDY loses the different ways each song appeals to the same listener. We also considered a prototype with learned dimension weights, but that still asked one point and one weighting to do too much. We chose to keep the library and let each candidate look for the parts relevant to it. A learned pooled representation remains a useful comparison.

We also had to decide how to tell the model that one song matters more to a listener than another. Just stick the score onto the audio vector? Multiply the whole vector by it? The April 9 design chose FiLM: let the score change how different dimensions are used, rather than turning the entire representation up or down uniformly. A separate attention bias gives higher-scored context songs more influence. Both mechanisms are in the code. Comparing them with the simpler options would test whether the extra complexity helps.

Within-library regression made a first task possible: reveal part of a library and predict transformed play counts for the remainder. The design preferred log normalization to compress large count differences while preserving ordering. It considered alternatives because old songs, background listening, and listener activity complicate count interpretation. The retained preprocessing and dataset-variant code implements several transforms and filters. The April 9 design also proposed per-listener linear regression as a cheaper way to test whether simple feature weights carried signal. That baseline was never completed.

We initially rejected sampled negatives because an absent song isn't a disliked song. It might be tomorrow's discovery that gets played fifty times. We hoped that learning relative preferences among known songs would carry over to unfamiliar ones. Applying the scorer to the wider catalog eventually forced us to confront how big a leap that was.

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

After the training sweep, we needed to actually listen to what this thing recommended. Predicting held-out scores wasn't the same as finding music somebody wanted to hear. The April 22 harness let us import a library, compare models, and listen. Without that step we could have kept celebrating a number.

A development checkpoint dated **April 22, 2026** reported an 18-trial sweep, with best validation Spearman **0.719** for a longer-training trial. It also reported a listening inspection in which the top recommendations were unfamiliar and strongly clustered in a narrow, slow instrumental/background style, and were judged unsatisfactory.

The checkpoint is the source for those results. Trial logs, evaluated weights, exact data/splits, recommendation lists, and listening judgments are absent. The code has changed since then, including the addition of hurdle heads, so this source snapshot does not reproduce that run.

The recommendations were bad. That's the part worth remembering alongside the number. Unfamiliar songs were the point, so not recognizing them wasn't the failure; listening to a narrow cluster of songs that didn't hit was. Our offline task had let us make progress on something that wasn't yet the experience we wanted.

## Competing interpretations

The April interpretation was **training-distribution collapse**: a narrow supervised signal could reward one audio region regardless of context. No controlled comparison establishes that as the cause. Other explanations remain plausible:

- Aggregate counts conflate exposure, availability, activity, habit, and preference; the objective may reward the wrong behavior.
- Within-library regression may extrapolate poorly to catalog discovery, while sampled negatives can introduce false negatives and sampling-prior effects.
- Coverage and fuzzy matching may distort the audio or library evidence presented to the model.
- Representation choice, optimization, capacity, implementation defects, or inference caching may affect results.
- Offline evaluation may reward correlations that do not improve listening satisfaction.

The dataset and negative-sampling scripts implement several of those comparisons. Their outcomes remain unresolved; the proposed linear baseline is still missing.

## After the listening mismatch: change the sample, or change the task?

The **April 22 retraining plan** prioritized changing the training distribution. Its reasoning was that catalog overlap did not establish useful supervision: finding familiar tracks in the embedding catalog did not show that training examples taught the model when to score related candidates highly. It considered inference-time filtering, reweighting existing examples, and collecting a more relevant cohort. The plan favored new collection, with reweighting as an intermediate diagnostic. This pursued the sampling diagnosis above; capacity, objective, and implementation problems remained open.

The **April 27 snowball plan** recorded a further decision, informed by a spot check: replace one uniform overlap filter with multiple tagged cohorts. The record argued that a popularity cap could exclude relevant musical regions and that modest overlap with a diverse library did not itself guarantee a broadly similar listener. It proposed stricter breadth-oriented overlap, looser region-filling collection, friend expansion, and long-tail seeds. Tagging cohorts would allow later training subsets to test different sampling theories without repeating collection. The plan records acceptance of that upfront collection tradeoff. The snowball and dataset-variant code implements these mechanisms. Which cohort helped, and whether stricter overlap actually encouraged breadth, remain open questions.

A **later negative-sampling and hurdle plan**, whose body has no reliable experiment date, revisited the April rejection of negatives. It diagnosed a different gap: the scorer had only seen targets already belonging to a listener's library. It proposed a fixed dataset and backbone, varying the number of random catalog targets to isolate the loss/target change. The hurdle variant separated membership discrimination from positive-score regression, intended to avoid a single MSE objective being dominated by many zero targets. Those variants are implemented; the sweep results are absent. Their labels still distinguish prepared-library membership from sampled absence, so the original concern about mistaking unheard songs for disliked songs remains.

These threads identify different uncertainties. Broader coverage might improve the available evidence; targeted cohorts might increase relevant supervision; new losses might change what the scorer learns to distinguish. None alone resolves whether the evaluation target captures satisfying discovery. The negative-sampling concern did not disappear when negatives were implemented: it became a tradeoff to measure. [Experiments](EXPERIMENTS.md) separates those mechanisms and proposes comparisons that could distinguish them.

## May and July: scale the thesis, first define the proof

The **May 20, 2026** discussion asked how bad an LLM would be with the equivalent of our model size and amount of data. Would it even make intelligible sentences? If not, how much should we conclude from this small music model being bad? Another recurring question was order: arbitrarily rearranging an aggregate library shouldn't change someone's taste. The order they actually listened to songs might tell us something different. That discussion sharpened the reasoning around scale and unordered attention; the April architecture was already in place.

Mo's **July 13 note** made the connection explicit: "You don't need to teach it the rules of grammar or the differences in languages, the NN will figure that out. So why wouldn't that be true for music?" Enough clean data, a simple but effective architecture, attention, and enough parameters to learn the connections. That motivated trying raw behavior and metadata instead of deciding what a play count should mean before the model saw it. The obstacle was getting enough correct data for a serious test.

**Marty's response was to suggest a narrower first test:** once we know someone has encountered a song, predict whether they come back and how much they replay it. That gets rid of the worst ambiguity around a song they've never heard. He proposed keeping count, recency, and metadata as explicit inputs and predicting future behavior from past information only. That experiment remains a proposal.

That proposal is different from the existing membership/score hurdle target. It requires an explicit time horizon, exposure definition, censoring policy, and time-correct features. A first observed play reduces exposure ambiguity but may not be the listener's true first encounter. Repeat behavior also remains affected by playlist placement and habit. Success on return prediction would support a narrower claim than discovery of never-before-played music.

Marty also proposed temporal holdouts, strong behavioral baselines, shuffled-audio controls, and a model/data scaling ladder. Those would help us find out whether the model was really learning from the music and whether more data and capacity were helping. [Experiments](EXPERIMENTS.md) lays out those tests for whoever picks this up next.
