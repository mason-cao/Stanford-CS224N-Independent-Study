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

## Part 2: Why Position Embeddings Are Necessary

The handout simplifies a Transformer block down to two main pieces:

```text
self-attention layer
feed-forward layer applied independently at each position
```

For input:

```text
X in R^{T x d}
```

where `T` is sequence length and `d` is hidden dimension, the attention projections are:

```text
Q = X W_Q
K = X W_K
V = X W_V
```

with:

```text
W_Q, W_K, W_V in R^{d x d}
Q, K, V in R^{T x d}
```

The self-attention output is:

```text
H = softmax(Q K^T / sqrt(d)) V
```

Then the feed-forward layer is:

```text
Z = ReLU(H W_1 + 1 b_1) W_2 + 1 b_2
```

where:

```text
W_1, W_2 in R^{d x d}
b_1, b_2 in R^{1 x d}
1 in R^{T x 1}
Z in R^{T x d}
```

The key question is: does this block know where a token appeared in the sentence?

Without position information, the answer is no. The block can compare tokens to other tokens, but it has no separate signal that says "this token was first" or "this token came after that token."

## Permutation Matrix Setup

A permutation matrix `P in R^{T x T}` reorders the rows of a sequence matrix. If:

```text
X_perm = P X
```

then `X_perm` contains the same token vectors as `X`, but in a shuffled order.

The handout asks me to show:

```text
Z_perm = P Z
```

This is called permutation equivariance. It does not mean the output is unchanged. It means the output changes in exactly the same way as the input was permuted. Shuffle the input rows, and the output rows are shuffled by the same permutation.

That is a problem for language because token order is not just a display choice. Word order changes grammatical roles and meaning.

## Self-Attention Under A Permutation

Start with the permuted input:

```text
X_perm = P X
```

The projected queries, keys, and values become:

```text
Q_perm = X_perm W_Q = P X W_Q = P Q
K_perm = X_perm W_K = P X W_K = P K
V_perm = X_perm W_V = P X W_V = P V
```

The score matrix for the permuted input is:

```text
Q_perm K_perm^T / sqrt(d)
```

Substitute the permuted projections:

```text
(P Q) (P K)^T / sqrt(d)
```

Use:

```text
(P K)^T = K^T P^T
```

so:

```text
(P Q) (P K)^T / sqrt(d)
= P Q K^T P^T / sqrt(d)
```

If:

```text
A = Q K^T / sqrt(d)
```

then:

```text
A_perm = P A P^T
```

The handout gives:

```text
softmax(P A P^T) = P softmax(A) P^T
```

So the attention-weight matrix becomes:

```text
S_perm = P S P^T
```

where:

```text
S = softmax(A)
```

Now compute the permuted attention output:

```text
H_perm = S_perm V_perm
```

Substitute:

```text
H_perm = (P S P^T) (P V)
```

Since:

```text
P^T P = I
```

the middle terms cancel:

```text
H_perm = P S V = P H
```

So the self-attention layer is permutation equivariant:

```text
X_perm = P X  implies  H_perm = P H
```

The attention layer did not detect that the input order was "wrong." It just carried the shuffle through.

## Feed-Forward Layer Under A Permutation

The feed-forward layer is applied to every position with the same weights. That weight sharing is useful, but it also means the feed-forward layer does not introduce position information by itself.

From the attention result:

```text
H_perm = P H
```

The first affine part of the feed-forward layer is:

```text
H_perm W_1 + 1 b_1
```

Substitute:

```text
P H W_1 + 1 b_1
```

A permutation matrix does not change the all-ones column:

```text
P 1 = 1
```

So:

```text
1 b_1 = P 1 b_1
```

Therefore:

```text
P H W_1 + 1 b_1
= P H W_1 + P 1 b_1
= P (H W_1 + 1 b_1)
```

The handout also gives:

```text
ReLU(P A) = P ReLU(A)
```

because permuting rows before a pointwise nonlinearity is the same as applying the pointwise nonlinearity and then permuting rows.

Now:

```text
Z_perm = ReLU(H_perm W_1 + 1 b_1) W_2 + 1 b_2
```

Substitute the previous expression:

```text
Z_perm = ReLU(P (H W_1 + 1 b_1)) W_2 + 1 b_2
```

Move the permutation through ReLU:

```text
Z_perm = P ReLU(H W_1 + 1 b_1) W_2 + 1 b_2
```

Use `1 b_2 = P 1 b_2`:

```text
Z_perm = P ReLU(H W_1 + 1 b_1) W_2 + P 1 b_2
```

Factor out `P`:

```text
Z_perm = P (ReLU(H W_1 + 1 b_1) W_2 + 1 b_2)
```

The expression inside the parentheses is exactly `Z`, so:

```text
Z_perm = P Z
```

This completes the proof for the simplified Transformer block.

## What The Proof Means In Plain Terms

The proof says the block treats the input as a set of token vectors with interactions, not as an ordered sentence. It can produce contextual token representations, but the context is orderless unless some extra signal encodes order.

For example:

```text
the dog chased the cat
the cat chased the dog
```

These sentences contain the same words but do not mean the same thing. A model without position information has no direct way to distinguish "dog before chased" from "dog after chased." It can compare token identities, but it cannot know which occurrence came first.

This matters even more for autoregressive language modeling. Predicting the next token is inherently positional. The model needs to know which tokens are earlier context and which token position is currently being predicted.

## Adding Position Embeddings

Position embeddings add a position-specific vector to each token embedding:

```text
X_pos = X + Phi
```

where:

```text
Phi in R^{T x d}
```

The row `Phi_t` represents position `t`. After this addition, the representation at a row contains both:

- token identity information from the word embedding
- position information from the position embedding

Now the same word at two different positions does not have exactly the same input vector. If the word embedding for `dog` is `x_dog`, then:

```text
position 2: x_dog + Phi_2
position 5: x_dog + Phi_5
```

Those are different vectors if:

```text
Phi_2 != Phi_5
```

This breaks the symmetry that made the earlier block permutation equivariant over bare token embeddings. The model can now learn different behavior based on relative and absolute position signals.

## Sinusoidal Position Embeddings

The handout gives fixed sinusoidal position embeddings:

```text
Phi(t, 2i) = sin(t / 10000^{2i/d})
Phi(t, 2i + 1) = cos(t / 10000^{2i/d})
```

where:

```text
t in {0, 1, ..., T - 1}
i in {0, 1, ..., d/2 - 1}
```

This means each pair of dimensions uses a sine and cosine wave with a different wavelength. Low-index dimensions change faster; higher-index dimensions change more slowly because the denominator is larger.

The position vector for a token is therefore not a single scalar position index. It is a pattern across many frequencies. That makes position a continuous geometric signal that can be added to embeddings and consumed by the same dot-product machinery as everything else.

The sine/cosine pairing is useful because relative shifts have structured effects. For a single frequency, knowing both:

```text
sin(t / scale)
cos(t / scale)
```

makes it possible to express shifted positions using angle-addition identities. That is one reason sinusoidal encodings are a natural fixed-position scheme.

## Can Two Position Embeddings Be The Same?

For the exact handout question, position embeddings for two different tokens can be the same if those tokens occur at the same position in separate sequences. Position embeddings encode position, not token identity.

Example:

```text
sentence A position 0: "the" gets Phi_0
sentence B position 0: "cat" gets Phi_0
```

The tokens are different, but the position embedding is the same because both are at position `0`.

Within one sequence, two different positions are not expected to share the same full sinusoidal vector over the normal sequence lengths used by the model. Individual dimensions can repeat because sine and cosine are periodic, but matching the entire vector across all frequencies at a different position would require all frequency pairs to line up at once.

The clean answer I should remember:

- same position in different sequences: yes, same position embedding
- different positions in the same ordinary-length sequence: no, not expected for the full vector
- individual sinusoidal dimensions: yes, they repeat periodically

## Part 2 Postmortem

The position-embedding section is really a symmetry argument. Without positions, self-attention and per-position feed-forward layers are permutation equivariant:

```text
X_perm = P X  implies  Z_perm = P Z
```

That is mathematically neat but linguistically wrong. Text is ordered, and many meanings depend on that order. Position embeddings give the model a way to distinguish equal token embeddings at different rows of the sequence matrix.

For implementation, the practical checks are:

- token embeddings and position embeddings must have the same hidden dimension before addition
- position IDs should run from `0` to `T - 1` for the current sequence length
- position embeddings must be sliced or indexed to match the actual input length
- the causal mask and position embeddings solve different problems: position embeddings identify order, while the causal mask prevents future-token leakage

The next checkpoint should move from the written math into the starter-code structure: `MLP`, `CausalAttention`, `DecoderBlock`, `Transformer.forward`, `Transformer.generate`, and `Transformer.get_loss_on_batch`.
