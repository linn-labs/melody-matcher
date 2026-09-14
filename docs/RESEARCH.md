# The thesis, the attempts, and why we stopped

*The "I" in this account is Mo, the person whose frustration started the project. "We" includes the work with AI collaborators. Marty helped write this account from those conversations and the project records.*

## I don't think we've run out of good music

I think Spotify and Apple Music recommendations suck. Not because every song should be a banger, but because I keep finding songs I'm absolutely in love with that were released years, sometimes decades, ago. They existed the whole time. I just never got pointed toward them.

That's why I have such a problem with "there's only so much music out there." Sure. But I haven't found all the music I love. Every old song I discover and obsess over is a very concrete example of something the recommendation system missed. It doesn't tell us exactly how to build a better system. It does tell me there's a lot left to find, and I don't buy that we've hit the ceiling.

My complaint is that recommendations so often feel like souped-up nearest-neighbor search: you like this sort of thing, here's something else in the same neighborhood. I suspect we're using the wrong map. "Danceability," energy, genre, and who else played a song are thin descriptions of why a particular piece of music gets under someone's skin. The failure I care about is a system that knows the neighborhood but keeps missing the house.

I don't have Spotify's or Apple's production architecture in front of me. "Souped-up nearest neighbors" is my diagnosis of the experience and the approach I want to get beyond, not a claim that their entire stack is one algorithm or that neither uses learned audio representations. The objection is to what the system understands about music and the listener, not to the existence of a nearest-neighbor lookup somewhere in the pipeline. Better search through the wrong representation still gets you the wrong songs faster.

## Music DNA

I love both HARDY's "Favorite Country Song" and Aretha Franklin's "Don't Play That Song." I can't give you a clean explanation for why. Country and soul don't get me very far, and putting those labels next to each other doesn't explain the connection either.

But I don't think my taste is random. I think there's something in the music's DNA that explains why both hit for me. It might be in a voice, a bit of tension, the way a phrase lands, or a combination of things I don't have words for. I don't want to decide in advance which of those explanations is right. The whole idea is to let the model find connections our descriptions miss.

Think about recommending music to a friend you know well. Sometimes you hear a song and immediately know: *they're going to love this*. You aren't calculating its danceability. You've built a representation of that person's taste from all the music you've heard them love, and you have a representation of the new song even if you're hopeless at explaining either one. Our internal neural network does something I want the artificial one to learn. That analogy is where my conviction comes from; it's not a neuroscience result.

By "DNA" I mean that underlying musical information, not a literal genetic code or a list of universal taste coordinates. We tried representing it with audio embeddings. Whether our chosen embeddings preserve enough of it is one of the questions the work has to answer. A vector is a container, not a guarantee that you've put the right thing inside it.

I believe a sufficiently capable model, trained on enough of the right listening behavior and rich enough representations of the music, can learn why apparently unrelated songs belong together for a particular person. We shouldn't have to teach it a set of genre rules or explain in words what makes a song hit. We should give it the evidence and the tools to make those connections.

This is the accumulation of work to test the thesis of a very opinionated 25-year-old. I can be wrong, and some of my ideas will be dumb. That doesn't mean I need to preface every idea with an apology. I think the thesis is right. We haven't made it work yet.

## More of this feeling, without playing the song to death

There's more to this than recommending a random song I might enjoy.

Sometimes I'm completely obsessed with one song. I want to keep getting whatever it gives me, but I don't want to listen to it so much that I ruin it for myself. What I'd like to ask is: *find me something else that scratches this particular itch*.

That should mean something closer in the parts of musical DNA that matter to me right now, not just another track with high danceability. In a learned space that actually captures that relationship, I should be able to stay close to the song I'm obsessed with or reach further out while still finding something I'll love. Those are different kinds of discovery, and both are interesting.

This also explains why I can criticize nearest-neighbor recommendations and still talk about closeness in vector space. Distance is only meaningful after you've learned what should count as close, for whom, and in relation to which song. I want the relationship learned from the music and the listener, rather than declared by a handful of descriptors.

The April design already suggested emphasizing one context song's score as a way to ask for "more like this." That's a starting idea, not a working control we can hand you. The current candidate scorer doesn't expose a proven music-DNA distance or an obsession slider. It's an attempt to learn some of the relationship we'd need first.

## Music is the first test

If we can make this work for music, I think the same logic could apply to nearly anything we'd want recommended to us. Books, movies, maybe even clothes. Represent the thing as a vector that captures something meaningful about it, then train a large neural network on people's preferences against those representations. Let it learn the connections instead of asking us to explain our taste first.

Two books can leave you with the same feeling without sharing a genre or a plot. Two movies can have almost nothing in common in a catalog and still be exactly your kind of movie. With clothes, you can sometimes look at something and immediately know it's you, even if you'd never have thought to search for it. That's the kind of relationship I want a model to learn. The descriptions we use to sort things aren't the whole reason we love them.

The representation and the evidence would change with the medium. A book gives us text; a film gives us images, dialogue, sound, and how they unfold together. Clothes bring in how something looks, feels, and fits. A purchase isn't automatically love, any more than a play is. We'd need to give each model the information that matters rather than assume that putting anything into a vector makes it useful.

But I don't think there's a wall around music here. If the approach succeeds, I want to take the same bet elsewhere: rich representations of the things themselves, enough real evidence of people's preferences, and a network capable of learning the relationship between them. Music is where we're trying to earn that confidence. Books, movies, and clothes are the ambition beyond it, not experiments we've already run.

## Why it's paused

We didn't pause because I stopped caring about the idea. I still run into the problem every time I discover an old song I should have heard years ago. We paused because we couldn't see a convincing route to the data needed to test it properly.

Our latest working hypothesis is mainly a data problem: we don't have the right user listening data, or enough of it, and the downstream network may already be too big for the evidence we can feed it. That's different from saying it's too capable for the problem. It can be too big for our dataset and still be nowhere near the scale needed to learn what we're asking it to learn.

The temptation is to make the model bigger. I share that temptation. But we can't scale the NN meaningfully without scaling the data, and we don't currently have a way to get the quality and volume we'd need. Access to listening behavior and usable song recordings, not another architecture diagram, is the constraint. A smaller model would still be a useful comparison; it wouldn't magically supply the missing evidence.

The problem kept turning up in different forms. Who did we collect? Which of their songs could we represent? Did lots of plays mean love, years of playlist placement, or simply more opportunities to hear the track? Was an absent song unwanted or just undiscovered? Was the model learning taste, or learning the shape of the sample we'd assembled? Changing the loss doesn't settle those questions, and collecting more of the same biased signal may just teach it the wrong thing more confidently.

That's our current explanation for why the work is stuck, not a measured verdict that model size caused the failure. The earlier records leaned hard toward a sampling diagnosis. The later conversations broadened the concern to the data/model balance and what we were even training the model to predict. We haven't run the controlled comparisons that would settle those competing explanations.

By July 13, I'd stepped back from the imagined product: forget whether someone imports a whole library, selects a few songs, or finds the interface convenient. What's the fastest path to proving the idea? The inputs still have to scale. Asking everyone to rate every song from one to ten isn't that path. I wanted to try audio embeddings and actual plays, add metadata where useful, and let the model learn what mattered instead of handing it a preference score we'd invented first.

I still want that experiment to happen. Releasing the work is a way of making our attempts and mistakes useful to somebody who might have the data, access, or better idea that we don't.

## What the records can tell you

The rest is the engineering trail: what we built, what the dated records say happened, and why we changed direction. The original data, trained weights, and trial logs aren't included, and we haven't rerun the experiments for this release. I'll keep the distinction between a result and a hunch where it matters. The thesis doesn't need a disclaimer after every sentence.

## How the approach took shape

### Early plan and March: obtain behavioral evidence and usable audio

We started with a fairly practical plan: pair a pretrained audio representation with Last.fm track counts, then train a two-tower model that summarizes a listener into one vector for retrieval. Counts were something we could actually collect without asking people to sit down and score their entire music library. They gave us a starting signal. The question was how much of taste that signal really captured.

We considered MERT, CLAP, OpenL3, and Jukebox, and chose MERT as the starting representation because it was music-focused and manageable to work with. Training our own audio encoder first would have made an already difficult project much harder. We didn't run a comparison proving MERT was the best choice; we picked a plausible starting point. The extractor still uses it. Its actual dimensions are listed below, since some early notes got those wrong.

The **March 3, 2026 checkpoint** described discovery aimed at listeners with broad tastes, followed by aggregate top-track collection and diversity filtering. The intention was to expose the model to connections across musical categories. The same record reported uneven discovery yield and a strong skew toward active listeners with deep histories. It recorded a decision to proceed with the collected pool rather than immediately repeat failed discovery attempts, pending the diversity results. Coverage was already a research constraint, not just a count of collected users.

The **March 5 checkpoint** separated two questions: durable taste from unordered libraries, and next-song context from chronological events. It described Deezer preview matching and timestamped collection infrastructure. The **March 24 checkpoint** then reported that saved preview URLs had expired before embedding extraction; matching and extraction were consolidated so previews could be resolved near use. That was a response to an acquisition failure, not evidence of a better taste representation. These dated reports do not establish current service availability or permission; [Data and rights](DATA_AND_RIGHTS.md) covers that separate boundary.

### April design: score each candidate against the library

By the **April 7 principles and April 9 scoring design**, we had rejected the single-vector starting point. A crude average of Aretha and HARDY isn't what I want to listen to. More generally, one center of gravity seemed like the wrong way to represent all the different ways music can appeal to one person. We also considered a prototype with learned dimension weights, but that still asked one point and one weighting to do too much. We chose to keep the library and let each candidate look for the parts relevant to it. That was our architectural bet; a strong learned pooled representation still deserves a proper comparison.

We also had to decide how to tell the model that one song matters more to a listener than another. Just stick the score onto the audio vector? Multiply the whole vector by it? The April 9 design chose FiLM: let the score change how different dimensions are used, rather than turning the entire representation up or down uniformly. A separate attention bias gives higher-scored context songs more influence. Both mechanisms are in the code. Whether they earn their complexity over the simpler options is still an experiment worth doing.

Within-library regression made a first task possible: reveal part of a library and predict transformed play counts for the remainder. The design preferred log normalization to compress large count differences while preserving ordering. It considered alternatives because old songs, background listening, and listener activity complicate count interpretation. The retained preprocessing and dataset-variant code implements several transforms and filters; their presence does not establish a winning target. An April 9 per-listener linear-regression baseline was also proposed as a cheaper probe of whether simple feature weights carried signal. It remained a proposal, not a completed comparison or proof that nonlinear attention was necessary.

We initially rejected sampled negatives for a reason I still think matters: if a song isn't in my library, you don't know that I dislike it. I might be about to discover it and play it fifty times. We hoped that learning relative preferences among known songs would carry over to unfamiliar ones. Applying the scorer to the wider catalog eventually forced us to confront how big a leap that was.

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

This is a dated historical report, not an independently verified benchmark or a reproducible listening study. Trial logs, evaluated checkpoints, exact data/splits, recommendation lists, and listening judgments are not distributed. The current source has evolved since that report, including hurdle-related changes; its presence cannot recreate the historical execution state. The report's claims of a working harness and a settled diagnosis are not adopted as current guarantees.

The recommendations were bad. That's the part worth remembering alongside the number. Unfamiliar songs were the point, so not recognizing them wasn't the failure; listening to a narrow cluster of songs that didn't hit was. Our offline task had let us make progress on something that wasn't yet the experience we wanted.

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

On **May 20, 2026**, I was asking how bad an LLM would be if it had the equivalent of our model size and amount of data. Would it even make intelligible sentences? If not, how much should we conclude from this small music model being bad? I also kept coming back to order: the arbitrary order of songs in an aggregate library shouldn't change someone's taste. The order they actually listened to songs might tell us something different. That discussion sharpened the reasoning around scale and unordered attention; the April architecture was already in place.

In my **July 13 note**, I put it this way: "You don't need to teach it the rules of grammar or the differences in languages, the NN will figure that out. So why wouldn't that be true for music?" I wanted enough clean data, a simple but effective architecture, attention, and enough parameters to make the connections. That was why I wanted to try raw behavior and metadata rather than deciding what a play count should mean before the model ever saw it. The obstacle was getting enough correct data to make that a serious test. Bigger models can't tell us about listening opportunities we never recorded just because we wish they could.

**Marty's response was to suggest a narrower first test:** once we know someone has encountered a song, predict whether they come back and how much they replay it. That gets rid of the worst ambiguity around a song they've never heard. He proposed keeping count, recency, and metadata as explicit inputs and predicting future behavior from past information only. We haven't adopted or run that experiment; it's one possible way to test the thesis without pretending to have solved all of discovery at once.

That proposal is different from the existing membership/score hurdle target. It requires an explicit time horizon, exposure definition, censoring policy, and time-correct features. A first observed play reduces exposure ambiguity but may not be the listener's true first encounter. Repeat behavior also remains affected by playlist placement and habit. Success on return prediction would support a narrower claim than discovery of never-before-played music.

Marty also proposed temporal holdouts, strong behavioral baselines, shuffled-audio controls, and a model/data scaling ladder. Those would help us find out whether the model was really learning from the music and whether more data and capacity were helping. [Experiments](EXPERIMENTS.md) lays out those possible tests, not a promised development roadmap. I'm putting them here because someone picking this up should get the questions as well as the code. I still think there's something worth finding in this.
