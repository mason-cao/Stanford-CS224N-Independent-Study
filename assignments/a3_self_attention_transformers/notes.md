# A3 Notes: Self-Attention And Transformers

## Sources

- Official A3 handout: <https://web.stanford.edu/class/cs224n/assignments_w26/a3.pdf>
- Official A3 code release: <https://web.stanford.edu/class/cs224n/assignments_w26/a3.zip>
- Week 3 Transformer slides: <https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture05-transformers.pdf>
- Week 3 self-attention notes: <https://web.stanford.edu/class/cs224n/readings/cs224n-self-attention-transformers-2023_draft.pdf>
- Previous backprop notes: <../../notes/lecture-notes/02_neural_networks_backprop.md>

## What This Assignment Is Testing

A3 is the first assignment where the course shifts from smaller neural classifiers into the architecture that dominates modern language modeling. The handout has three parts:

- attention exploration: reason through query-key-value attention with simple vector geometry
- position embeddings: prove why a bare Transformer block is insensitive to token order except for the same permutation being carried through the output
- implementation: fill in a decoder-only Transformer, test each module, compute language-model loss, and run a short training loop

I should connect all three parts. The written questions explain why the code has the shape it has. `CausalAttention.forward` is not just a PyTorch exercise; it is the vectorized version of the query-key-value weighted average. `Transformer.forward` is not just stacking layers; it is where token embeddings, position embeddings, causal attention, feed-forward blocks, normalization, and output logits become one autoregressive language model.

## Checkpoint Map

I am splitting A3 into three clean checkpoints:

1. Attention mechanics and multi-head intuition.
2. Position embeddings and permutation-equivariance proof.
3. Transformer implementation notes and local testing plan.

This checkpoint is only the first one. I am not pulling the starter archive into the repo yet, because the existing project pattern is to document the assignment without mirroring Stanford starter code.

## Part 1: Attention As A Soft Lookup

The attention operation in the handout uses:

```text
q in R^d
k_i in R^d for i = 1, ..., n
v_i in R^d for i = 1, ..., n
```

The query `q` asks a question. Each key `k_i` is compared with the query using a dot product:

```text
score_i = k_i^T q
```

The scores become a probability distribution through softmax:

```text
alpha_i = exp(k_i^T q) / sum_j exp(k_j^T q)
```

The output is a weighted average of the value vectors:

```text
c = sum_i alpha_i v_i
```

So the attention layer is not choosing one value with a hard index. It makes a soft lookup. The query scores every key, softmax turns those scores into weights, and the output mixes values according to those weights.

The lookup-table analogy is useful but incomplete:

- in an ordinary lookup table, a key either matches or does not match
- in attention, every key matches to some degree
- the values are not returned separately; they are averaged into one vector

That last point is easy to overlook. Attention returns a convex combination of value vectors because the softmax weights are nonnegative and sum to one.

## Copying A Value Vector

Attention can copy a value when one key-query score is much larger than all the others. If one index `j` satisfies:

```text
k_j^T q >> k_i^T q for every i != j
```

then:

```text
alpha_j ~= 1
alpha_i ~= 0 for i != j
```

and the output becomes:

```text
c ~= v_j
```

The exponential in softmax is the reason the gap matters so much. A small score advantage can become a larger probability advantage, and a very large score advantage makes the distribution nearly one-hot.

The geometric condition is that `q` points strongly in the same direction as `k_j`, or at least has a much larger projection onto `k_j` than onto the other keys. If the keys are normalized, this is mostly about direction. If the keys are not normalized, both direction and norm affect the dot product.

## Why Dot Products Measure Matching

For two vectors:

```text
k_i^T q = ||k_i|| ||q|| cos(theta_i)
```

where `theta_i` is the angle between them. The score is large when:

- the vectors point in similar directions
- one or both vectors have large norm

This explains a later pitfall in the handout. If key norms vary a lot, dot-product attention can change even when the key direction is basically the same. A larger key norm makes the same directional match produce a bigger score.

That is why normalization choices matter. A dot product does not only ask "are these directions aligned?" It also asks "how large are these vectors?"

## Averaging Two Values With One Head

Suppose all keys are orthogonal and have norm one:

```text
k_i^T k_j = 0 for i != j
||k_i|| = 1
```

If I want attention to average `v_a` and `v_b`, a natural query is:

```text
q = beta (k_a + k_b)
```

for a large positive scalar `beta`.

Then the scores are:

```text
k_a^T q = beta
k_b^T q = beta
k_i^T q = 0 for i not in {a, b}
```

The two desired keys receive equal high scores, while all other orthogonal keys receive score zero. As `beta` gets large, softmax puts almost all probability mass on `a` and `b`, split equally:

```text
alpha_a ~= 1/2
alpha_b ~= 1/2
alpha_i ~= 0 for i not in {a, b}
```

So:

```text
c ~= (v_a + v_b) / 2
```

This is the clean mathematical story. The practical issue is that it depends on balancing scores inside one softmax distribution. If the scores for `a` and `b` stop being equal, the output stops being a stable average.

## Small Key Noise

The handout then replaces exact keys with random keys:

```text
k_i ~ N(mu_i, Sigma_i)
```

with mutually orthogonal unit means:

```text
mu_i^T mu_j = 0 for i != j
||mu_i|| = 1
```

If the covariance is tiny and isotropic:

```text
Sigma_i = alpha I
```

with very small `alpha`, then the sampled key `k_i` stays close to `mu_i` in every direction. The earlier query idea still works if I use the means:

```text
q = beta (mu_a + mu_b)
```

The reason is that:

```text
k_a^T q ~= mu_a^T q = beta
k_b^T q ~= mu_b^T q = beta
k_i^T q ~= mu_i^T q = 0 for i not in {a, b}
```

Tiny isotropic noise perturbs the scores, but not enough to change the main softmax pattern when `beta` is chosen sensibly. The output should stay near:

```text
(v_a + v_b) / 2
```

This is an important distinction: single-head attention is not automatically fragile. It is reasonably stable under small random perturbations. The harder case is structured variation that changes one desired score much more than the others.

## Larger Norm Variation In One Key

The handout's harder case is when `k_a` still points roughly in the `mu_a` direction but its magnitude varies a lot. A simplified mental model is:

```text
k_a ~= r mu_a
```

where `r` changes across samples. With:

```text
q = beta (mu_a + mu_b)
```

the score for key `a` becomes:

```text
k_a^T q ~= r beta
```

while the score for key `b` stays near:

```text
k_b^T q ~= beta
```

Now the softmax balance between `a` and `b` depends on the random magnitude `r`. If `r > 1`, `a` can receive too much attention. If `r < 1`, `a` can receive too little. Because softmax is exponential, the change in attention weight can be sharper than the change in norm.

Qualitatively, the output `c` varies along the space between the desired average and whichever value receives too much probability. The intended behavior is a stable average:

```text
(v_a + v_b) / 2
```

but the sampled outputs can lean toward:

```text
v_a
```

or toward:

```text
v_b
```

depending on how the unstable score changes. The problem is not that one value can never be copied. The problem is that one head is trying to maintain a precise ratio between two competing softmax weights, and norm variation disturbs that ratio.

## Multi-Head Attention As A Cleaner Decomposition

The multi-head setup in the handout uses two queries and averages their outputs:

```text
c_final = (c_1 + c_2) / 2
```

Instead of asking one head to split attention between `a` and `b`, I can let each head specialize:

```text
q_1 = beta mu_a
q_2 = beta mu_b
```

Then:

```text
c_1 ~= v_a
c_2 ~= v_b
```

and:

```text
c_final ~= (v_a + v_b) / 2
```

This is a better decomposition. Each head only has to make one item dominate its own softmax distribution. It does not need to preserve an exact 50/50 balance between two high-scoring keys inside one distribution.

When `k_a` has high magnitude variance, the first head can still have some variance in how sharply it attends to `a`. But the second head, which targets `b`, is not forced to compete against the unstable `a` score in the same softmax. The final output averages two simpler retrievals instead of depending on one delicate mixed retrieval.

The lesson is not merely "more heads means more capacity." The sharper point is:

```text
one head mixes information by one probability distribution
multiple heads can perform multiple simpler lookups and combine the results afterward
```

That gives the model a more stable way to represent several relationships at once.

## Connection To The Transformer Code

The implementation part will vectorize the same idea. For a batch of token representations:

```text
X in R^{B x T x d_model}
```

the attention module creates:

```text
Q = X W_Q
K = X W_K
V = X W_V
```

For one head, the attention score matrix is:

```text
scores = Q K^T / sqrt(d_head)
```

Shape check for one batch item and one head:

```text
Q: T x d_head
K: T x d_head
V: T x d_head
scores: T x T
attention weights: T x T
output: T x d_head
```

Each row of `scores` answers: for this query token, how much should it attend to every key token?

The decoder-only version must also apply a causal mask. Without the mask, token position `t` could attend to future positions and leak target information during language-model training. With the mask, row `t` can only attend to columns `0` through `t`.

The causal mask is therefore not a modeling decoration. It is what makes next-token prediction honest.

## Part 1 Postmortem

The central mental model is:

```text
attention = softmax(query-key scores) applied to values
```

Copying works when one score dominates. Averaging with one head works on paper when two scores are equal and much larger than the rest, but that balance can be fragile when key norms vary. Multi-head attention avoids some of that fragility by letting separate heads perform separate focused lookups before combining their outputs.

For implementation, the likely mistakes are:

- forgetting that attention weights live over source positions, not hidden dimensions
- applying softmax over the wrong axis
- forgetting the `1 / sqrt(d_head)` scaling
- building a causal mask with the wrong triangular direction
- accidentally allowing a decoder token to attend to future tokens

The next checkpoint should handle why position information has to be added before attention. Without position embeddings, the same set of token vectors shuffled into a different order just produces the same output vectors shuffled in the same way.
