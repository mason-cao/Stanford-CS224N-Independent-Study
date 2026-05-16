# A1: Introduction to Word Vectors

## Official Slot

- Course week: Week 1
- Course schedule: <https://web.stanford.edu/class/cs224n/#schedule>
- Official A1 code release: <https://web.stanford.edu/class/cs224n/assignments_w26/a1.zip>

## Repo Status

In progress.

I am working through the word-vector material and keeping notes before doing any larger implementation work.

## What A1 Is Actually Asking Me To Learn

A1 is not just a warmup notebook. It forces me to connect the lecture story about word-vector geometry to concrete operations:

- build a vocabulary from tokenized documents
- count local co-occurrences inside a fixed window
- reduce a sparse count matrix with SVD
- compare that count-based geometry against pretrained GloVe vectors
- use cosine similarity to inspect neighbors, analogies, polysemy, antonymy, and bias

The important thing is that every plotted point or nearest-neighbor list comes from a modeling choice. If the result looks strange, I should ask which choice created the behavior: corpus domain, preprocessing, window size, count matrix construction, SVD projection, pretrained corpus, vector dimensionality, or the cosine metric.

## Notebook Structure

The official A1 notebook is `exploring_word_vectors.ipynb`. It has two main parts:

- Part 1: count-based word vectors using the Large Movie Review dataset.
- Part 2: prediction/count-hybrid vectors using pretrained GloVe embeddings.

Part 1 is more implementation-heavy. Part 2 is more analysis-heavy. That seems intentional: first I build a small version of the machinery, then I use a larger pretrained representation and look for both strengths and failures.

## Part 1 Implementation Notes

### `distinct_words`

Input shape:

```text
corpus = [
  [token_1, token_2, ...],
  [token_1, token_2, ...],
  ...
]
```

Output:

```text
corpus_words, n_corpus_words
```

The subtle point is that this function should not do extra normalization. The notebook's `read_corpus` lowercases and strips punctuation for the movie review data, but the toy tests preserve tokens like `All` and `All's`. If I lowercase or clean inside `distinct_words`, I will make the function less general and likely break the sanity check.

The clean mental model:

- flatten all token lists
- take a set to remove duplicates
- sort the resulting word types
- return both the sorted list and its length

Sorting matters because later matrix rows and columns need stable indices. Without a deterministic vocabulary order, the co-occurrence matrix can have the right counts but the wrong row labels.

### `compute_co_occurrence_matrix`

For a fixed window size `n`, every token becomes a center word once. If the center position is `i`, the context positions are:

```text
max(0, i - n) through min(len(document), i + n + 1)
```

excluding `i` itself.

For each center-context pair, increment:

```text
M[word2ind[center], word2ind[context]] += 1
```

The matrix becomes symmetric because each neighboring pair is eventually seen from both directions. For example, in a window of 1, `<START>` counts `All` as context when `<START>` is the center, and `All` counts `<START>` as context when `All` is the center.

Things to be careful about:

- words near document edges have smaller context windows
- repeated words should contribute repeated counts
- `<START>` and `<END>` are real tokens for this assignment
- `word2ind` must match the sorted order from `distinct_words`

### `reduce_to_k_dim`

The notebook wants truncated SVD, not a hand-written decomposition. The goal is to turn:

```text
M shape: (vocab_size, vocab_size)
```

into:

```text
M_reduced shape: (vocab_size, k)
```

Using `TruncatedSVD` returns something like `U * S`, not just `U`. That means the resulting coordinates include singular-value scaling. The assignment later normalizes rows before plotting, which shifts the plot toward directional similarity rather than raw length.

I should not overinterpret the exact 2D plot. SVD is compressing a high-dimensional count matrix into two axes, so the picture is a lossy diagnostic. Clusters are worth explaining, but a bad-looking 2D projection does not necessarily mean the full representation has no useful structure.

### `plot_embeddings`

This is mainly a sanity check that I can map word labels to rows:

```text
row = M_reduced[word2ind[word]]
x = row[0]
y = row[1]
```

The plot should show only the requested words, not the whole vocabulary. Label placement matters because the written analysis depends on visually comparing a small set of points.

## Part 1 Analysis Notes

The co-occurrence plot uses movie reviews, so I should expect the geometry to reflect that domain. Words like `movie`, `story`, `book`, `good`, and `interesting` may be shaped by review language rather than general English.

When answering cluster questions, I need to distinguish at least three possible explanations:

- semantic similarity: words mean similar things
- topical association: words appear in the same kind of review even if they are not synonyms
- projection artifact: words look close in 2D because the projection lost information

That third point matters. If `large`, `massive`, and `huge` do not cluster cleanly in the count-based plot, it might not mean the corpus failed to notice size adjectives. It could mean the two-dimensional SVD projection is too compressed, the sample is small, or those words are used in different review contexts.

## Part 2 Analysis Notes

GloVe changes the scale of the experiment. Instead of using a small co-occurrence matrix from 150 movie reviews, the notebook loads 200-dimensional pretrained vectors from a much larger corpus. This should generally produce cleaner semantic neighborhoods, but it also imports the biases and frequency patterns of the pretraining data.

### Cosine Similarity

Cosine similarity measures angle, not length:

```text
cosine(p, q) = (p dot q) / (||p|| ||q||)
```

That makes it useful for word vectors because vector length can encode frequency or confidence-like effects that I may not want to dominate nearest-neighbor lookup. If two words point in a similar direction, cosine similarity is high even if one vector has a larger norm.

Cosine distance is:

```text
1 - cosine_similarity
```

So smaller cosine distance means more similar under this metric.

### Polysemy

A static word vector gives one vector per word type. That creates an obvious failure mode for words with multiple meanings. The vector for a word like `bank`, `pitch`, or `crane` has to average across different contexts unless one meaning dominates the corpus.

If a top-10 neighbor list only shows one sense, that does not prove the other sense is absent from English. It may mean:

- one sense is much more frequent in the training corpus
- the less frequent sense has weaker or noisier neighbors
- the tokenization misses multiword expressions that would reveal the sense
- nearby words from different senses are not close enough to survive the top-10 cutoff

This is a direct limitation of static embeddings and a setup for why contextual word representations matter later.

### Synonyms And Antonyms

Distributional similarity is not the same thing as logical sameness. Antonyms often appear in very similar contexts:

```text
hot day / cold day
high price / low price
good movie / bad movie
```

Because the surrounding words overlap, cosine similarity can put antonyms close together. A synonym may be less close if it belongs to a different register, genre, or frequency band. This is why "near in embedding space" should usually mean "used similarly," not "means the same thing."

### Analogies

The analogy pattern is vector offset matching. For an analogy:

```text
x : y :: a : ?
```

the target direction is:

```text
y - x + a
```

The nearest word to that target vector is treated as the answer. This only works when the relation is represented as a reasonably consistent linear offset. That is a strong assumption. It works surprisingly often for some syntactic and semantic relations, but the notebook's incorrect-analogy examples are a reminder that nearest-neighbor search has no built-in understanding of the category I expected.

When an analogy fails, plausible reasons include:

- the intended relation is not linear in this embedding space
- the nearest word is a corpus-frequency artifact
- the input words have multiple meanings
- the model finds a topical association instead of the intended relation
- the vocabulary contains odd tokens that happen to be close to the arithmetic target

### Bias

The bias questions are not separate from the rest of the assignment. They are another consequence of the same distributional training logic. If the corpus repeatedly places demographic terms near different professions, traits, or social roles, those patterns can become directions in the vector space.

A useful way to think about this:

- word vectors preserve corpus regularities
- some corpus regularities reflect real usage
- some real usage reflects social stereotypes or historical exclusion
- downstream systems can reuse those patterns unless they are measured and mitigated

A mitigation strategy has to define what structure should be preserved and what structure should be reduced. For example, a debiasing method might identify a gender direction using definitional pairs, then reduce that component for words that should be neutral with respect to gender. The hard part is not just the linear algebra; it is deciding which associations are legitimate, which are harmful, and how to avoid hiding bias instead of removing it.

## Working Rules For My A1 Attempt

- Pass every provided sanity check before writing analysis answers.
- Keep implementation functions general; do not hard-code the toy tests.
- Treat 2D plots as diagnostics, not ground truth.
- For every nearest-neighbor or analogy answer, explain the corpus/context reason, not just the observed output.
- Save the exact examples I try, including failures, because the failed searches are usually more informative than the one that works.

## What Should Eventually Live Here

- starter materials after I pull them locally
- derivation notes specific to the assignment
- code experiments and debugging notes
- a short postmortem on what actually took time
