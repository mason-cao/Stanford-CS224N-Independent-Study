# A1: Introduction to Word Vectors

## Official Slot

- Course week: Week 1
- Course schedule: <https://web.stanford.edu/class/cs224n/#schedule>
- Official A1 code release: <https://web.stanford.edu/class/cs224n/assignments_w26/a1.zip>
- Public notebook preview: <https://web.stanford.edu/class/cs224n/assignments/a1_preview/exploring_word_vectors.html>

## Repo Status

In progress. Part 1 count-based vector implementation notes are complete; Part 2 GloVe analysis is next.

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

## Part 1 Finished Notes: Count-Based Vectors

This is the part of A1 where the representation is completely transparent. There is no trained neural model yet. Every coordinate starts as a count of how often one word appears near another word. That makes Part 1 a useful baseline because I can inspect every modeling choice directly.

### Implementation I Want To Remember

The implementation has three important invariants:

- the vocabulary order comes from `sorted(set(flattened_tokens))`
- the same `word2ind` mapping is used for both rows and columns
- context windows are clipped at document boundaries instead of padded

The core loop for the co-occurrence matrix is:

```python
for document in corpus:
    for center_index, center_word in enumerate(document):
        left = max(0, center_index - window_size)
        right = min(len(document), center_index + window_size + 1)

        for context_index in range(left, right):
            if context_index == center_index:
                continue
            context_word = document[context_index]
            M[word2ind[center_word], word2ind[context_word]] += 1
```

The matrix is symmetric if the window is symmetric and every token gets its own turn as the center word. I should not enforce symmetry manually with `M = M + M.T`, because that can double counts if the loop already visits both directions. The right test is whether the counting procedure creates the expected entries from the toy corpus.

For SVD, the assignment wants a reduced matrix with shape:

```text
number_of_words x k
```

`TruncatedSVD(n_components=k).fit_transform(M)` gives one row per word in the original row order. That is exactly what `plot_embeddings` needs. The returned coordinates include singular-value scaling, so nearby points in the plot are not pure rows of `U`; they are the projected count vectors in the reduced space.

### Toy-Corpus Sanity Check

The toy corpus is not just a throwaway test. It catches three common mistakes:

1. If I lowercase or strip punctuation inside `distinct_words`, then `All` and `All's` will not match the expected vocabulary.
2. If I forget to sort the vocabulary, the matrix counts can be right but assigned to the wrong row labels.
3. If I use the wrong window endpoints, edge words like `<START>` and `<END>` will have incorrect counts.

For a window size of 1, the pair `<START>, All` appears in both toy documents. The matrix entry:

```text
M[word2ind["<START>"], word2ind["All"]]
```

should therefore be 2. The reverse entry should also be 2 because each `All` later becomes the center word and sees `<START>` in its own window. This is the simplest way to check that I understand why the matrix is symmetric.

### What The 2D SVD Plot Can And Cannot Show

The 2D plot is a compressed diagnostic, not the actual high-dimensional representation. If two words are close in the plot, the safest interpretation is:

```text
after reducing this particular co-occurrence matrix to two dimensions,
these words have similar projected count patterns
```

That is weaker than saying the words are synonyms. A word can land near another word because they share sentiment-review contexts, topic contexts, syntactic contexts, or projection artifacts.

For the A1 word list, I expect at least these pressures:

- `movie`, `book`, and `story` can cluster because reviews often discuss narrative objects and media.
- `good`, `interesting`, and `fascinating` can cluster because they work as evaluative adjectives.
- `large`, `massive`, and `huge` might not cluster cleanly even though they are semantically related, because the small movie-review sample may use them in different contexts or too rarely.
- `mysterious` may behave more like a genre or tone word than a generic adjective, so its neighbors depend heavily on corpus domain.

The lesson is that count-based embeddings expose domain effects quickly. If the corpus is movie reviews, the representation is about how words behave in movie reviews, not how they behave in all of English.

### Part 1 Postmortem

What feels solid now:

- I can explain why row and column order have to be deterministic.
- I can derive the window boundaries without special-casing document edges.
- I understand why the co-occurrence matrix becomes symmetric under this setup.
- I know why row-normalizing the 2D output changes the plot toward directional comparison.

What I still need from Part 2:

- compare this small count-based geometry against pretrained GloVe vectors
- find examples where cosine similarity reflects usage rather than synonymy
- explain why static embeddings struggle with polysemy
- connect the bias questions back to distributional training data instead of treating them as a separate topic

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
