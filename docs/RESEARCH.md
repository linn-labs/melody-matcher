# The thesis and why it is paused

[Research history](HISTORY.md) · [Experiment index](EXPERIMENTS.md)

Better recommendations should learn the connections that make music matter to a particular person, including connections that genre labels and ordinary descriptions miss. That is the constructive goal behind Melody Matcher.

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

## Follow the learning history

The research changed direction several times: a single taste vector gave way to candidate scoring; a promising validation result failed the listening test; a sampling diagnosis led to revised snowball cohorts; the limits of positive-only targets brought negatives back; and the later discussion questioned the target and scale of the entire test.

**[Read the chronological account](HISTORY.md)** for the actual chain of beliefs, attempts, observations, and pivots. It preserves mistakes in our reasoning without turning every historical diagnosis into a proven cause. The [experiment index](EXPERIMENTS.md) links each stage directly to its detailed explanation.

The sections below describe the retained implementation, not a claim that every historical trial used precisely this final source.

## Implemented architecture

The authoritative local definitions are [model.py](../library/src/model.py), [layers.py](../library/src/layers.py), and [config.py](../config.py).

1. The selected external representation is MERT-v1-330M, configured for **1024-dimensional embeddings** and 24 kHz audio. The [extractor](../src/embedder.py) mean-pools the last hidden state over time and L2-normalizes it. Model weights are absent.
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

## What remains unresolved

The April 22 checkpoint reported an 18-trial sweep and best validation Spearman of 0.719, followed by recommendations that sounded bad: a narrow cluster of slow instrumental/background music that did not match the listener. The [history of that test](HISTORY.md#4-april-22-the-number-looked-promising-the-music-was-bad) explains why it changed the direction of the work. The original result artifacts are absent; no experiments were rerun for this snapshot.

Training-distribution collapse was the working diagnosis, not an established cause. Counts can reward exposure and habit; positive-only regression can extrapolate poorly; sampled negatives can invent rejection; matching and representation can lose important information; capacity and optimization can also matter. The subsequent [sampling changes](SAMPLING.md) and [target experiments](TRAINING_EXPERIMENTS.md) address different parts of this problem. No supplied result ledger identifies a winning remedy.

Moritz's July reset proposed learning from actual behavior and metadata instead of deciding beforehand what every count should mean. Marty's response proposed predicting return and replay after an observed encounter. That [possible next proof](FUTURE_PROOF.md) is distinct from the current hurdle implementation and remains unadopted and unexecuted.
