# Week 1 Notes: Word Vectors

## Sources

- CS224N schedule and Week 1 materials: <https://web.stanford.edu/class/cs224n/#schedule>
- Official public CS224N playlist: <https://www.youtube.com/playlist?list=PLoROMvodv4rOaMFbaqxPDoLWjDaRAdP9D>

## What Actually Clicked For Me

The main conceptual jump is that a word vector is not "meaning" in the abstract. It is a trainable compromise that becomes useful because the model is forced to predict context. That makes the geometry downstream of the objective function, not some magical semantic lookup table.

I also think it is easy to say "distributional hypothesis" without really cashing it out. In practice, the useful claim is narrower: words that appear in similar local environments end up with related representations because the model can reuse parameters across contexts. The embedding space is a side effect of that compression pressure.

## The Core Idea I Want To Keep Straight

The main idea is to stop treating words as isolated symbols. A one-hot vector can identify a word, but it does not say anything about how that word relates to other words. A dense vector is more useful because distance and direction can start to carry information about usage.

The part I want to keep clear is that the geometry is learned from context. Words end up near each other because the training signal rewards similar behavior in similar environments. That does not mean the vector "knows" meaning in a human sense, but it gives the model a much better starting point than raw word IDs.

I also want to be careful not to overread pretty embedding examples. Analogies and nearest neighbors are useful checks, but they are not the same thing as understanding the whole representation.

## Why One-Hot Vectors Are A Dead End

A one-hot vector is useful as an index, not as a representation. If the vocabulary has `|V|` words, every word gets a `|V|`-dimensional vector with one nonzero entry. That makes lookup easy, but it also means every pair of distinct words has dot product zero. In that representation, `hotel` and `motel` are no closer than `hotel` and `banana`.

The deeper issue is not just sparsity. The model has no built-in way to share statistical strength across related words. If I learn something from sentences containing `motel`, a pure one-hot representation gives me no reason to transfer that behavior to `hotel`. Any similarity has to be supplied externally by a resource like a thesaurus or WordNet, and those resources are brittle: they miss new usages, flatten context, and make similarity feel like a manually curated property rather than something learned from language data.

A dense embedding fixes this by making each dimension part of a shared coordinate system. The vector no longer only says "which word is this?" It gives the model trainable degrees of freedom that can be reused across many prediction contexts.

## Distributional Semantics In Operational Terms

The distributional hypothesis is easy to state and easy to misunderstand. The operational version for this course is:

> A word representation should be shaped by the local contexts in which the word appears.

For Word2Vec, context is usually a symmetric window around a center word. If the sentence fragment is:

```text
problems turning into banking crises as
```

and the center word is `banking` with window size 2, then the outside/context words are `turning`, `into`, `crises`, and `as`. The learning problem is not to define `banking` directly. The learning problem is to make the vector for `banking` useful for predicting the words that tend to surround it.

That is the important mental shift: meaning is not hand-entered into the vector. Meaning-like structure appears because the training task repeatedly rewards the model for placing words with similar context behavior in compatible parts of the space.

## Skip-Gram Word2Vec Objective

The Skip-gram setup predicts outside words from a center word. Given a corpus sequence:

```text
w_1, w_2, ..., w_T
```

and a context window size `m`, each position `t` creates prediction targets for the words near `w_t`:

```text
w_{t-m}, ..., w_{t-1}, w_{t+1}, ..., w_{t+m}
```

The likelihood multiplies the probabilities assigned to all real center-context pairs:

```text
L(theta) = product over t product over -m <= j <= m, j != 0
           P(w_{t+j} | w_t; theta)
```

The training objective is the average negative log likelihood:

```text
J(theta) = -(1 / T) sum over t sum over -m <= j <= m, j != 0
           log P(w_{t+j} | w_t; theta)
```

So minimizing `J(theta)` is the same basic goal as maximizing the probability of observed context words. The loss is high when the model gives low probability to real outside words.

## Two Vectors Per Word

Word2Vec uses two embeddings for each vocabulary item:

- `v_w`: the vector for word `w` when it appears as the center word
- `u_w`: the vector for word `w` when it appears as an outside/context word

For a center word `c` and an outside word `o`, the score is a dot product:

```text
score(o, c) = u_o^T v_c
```

A larger dot product means the model thinks `o` is a more plausible context word for `c`. To turn all vocabulary scores into a probability distribution, Word2Vec uses softmax:

```text
P(o | c) = exp(u_o^T v_c) / sum_{w in V} exp(u_w^T v_c)
```

This denominator matters conceptually because every candidate outside word competes against every other candidate. The model is not just asking "are `o` and `c` compatible?" It is asking "how compatible is `o` with `c` compared with all other words in the vocabulary?"

For one observed pair `(c, o)`, the loss is:

```text
J = -log P(o | c)
  = -u_o^T v_c + log sum_{w in V} exp(u_w^T v_c)
```

The first term rewards a high score for the true outside word. The second term penalizes scores spread across the whole vocabulary. This is why the model has to learn contrastive structure, not just make every dot product large.

## Gradient Intuition

For a single center-context pair, define `y` as the one-hot vector for the true outside word and `p` as the softmax distribution over all outside words. If `U` is the matrix of outside vectors, then the gradient with respect to the center vector is:

```text
dJ / dv_c = U^T (p - y)
```

This compact formula is worth unpacking:

- If the true outside word has probability too low, `(p - y)` is negative in that position, so the update pulls `v_c` toward `u_o`.
- If an incorrect outside word has probability too high, `(p - y)` is positive in that position, so the update pushes `v_c` away from that outside vector.
- The update is weighted by the model's own mistakes. Very implausible words that already have tiny probability barely matter for that example.

For an outside vector `u_w`, the gradient has the same error signal:

```text
dJ / du_w = (p_w - 1[w = o]) v_c
```

So the true outside vector is pulled toward the center vector, while incorrect outside vectors are pushed away in proportion to how much probability the model assigned them. This is the core mechanical reason similar words end up near each other: they participate in similar prediction errors across many local windows.

## Why The Full Softmax Becomes Expensive

The clean softmax objective is expensive because the denominator sums over the entire vocabulary for every training pair. With a large vocabulary, one update wants scores for every possible outside word, even though only one outside word was observed in the current pair.

That computational bottleneck motivates approximations like negative sampling. The idea I want to keep separate for now:

- Full softmax asks the model to rank the true outside word against the whole vocabulary.
- Negative sampling turns the problem into distinguishing observed center-context pairs from a small set of sampled noise words.

The approximation changes the training computation, but the representation-learning pressure is related: observed pairs should become more compatible, and sampled non-observed pairs should become less compatible.

## Things I Do Not Fully Trust Yet

- I understand the high-level distributional idea better now, but I still need to work through the exact gradient derivations by hand without leaning on the compact matrix form.
- I know analogies are a neat qualitative demo, but I do not yet trust them as serious evidence that an embedding space is "good."
- I want to run nearest-neighbor inspections and see where they fail, especially for polysemous words where one static vector has to collapse multiple senses.
- I need to study GloVe carefully enough to explain what changes when the objective starts from global co-occurrence counts instead of local prediction windows.

## Why This Matters For The Rest Of The Course

This week sets up a pattern that keeps showing up later:

- define a prediction problem
- learn dense vectors because discrete symbols do not compose well
- let the training objective shape the representation geometry

That logic does not stop with static word vectors. It scales all the way into contextual encoders and autoregressive language models.

## Repo Connection

This is the first set of notes I am using for A1. I have not started the later assignments yet.
