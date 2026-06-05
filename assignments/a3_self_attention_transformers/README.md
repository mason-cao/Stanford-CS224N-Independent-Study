# A3: Self-Attention and Transformers

## Official Slot

- Course week: Week 3
- Official handout: <https://web.stanford.edu/class/cs224n/assignments_w26/a3.pdf>
- Official code release: <https://web.stanford.edu/class/cs224n/assignments_w26/a3.zip>

## Repo Status

First study pass complete. Implementation sanity-check companion added.

I have completed a first study pass through the A3 written material and starter-code structure: attention exploration, position embeddings, and the decoder-only Transformer implementation map.

The first checkpoint is focused on attention as a soft lookup operation: how queries score keys, how softmax converts scores into weights, why copying a value is easy when one score dominates, why averaging values with one head is fragile, and why multiple heads give a cleaner way to compose information.

The second checkpoint is focused on why position information is mathematically necessary. Without position embeddings, the simplified Transformer block in the handout is permutation equivariant: if the input tokens are shuffled, the output is shuffled in exactly the same way. Position embeddings break that symmetry by making identical token embeddings different at different sequence positions.

The third checkpoint is focused on the implementation path through the official starter code: `MLP`, `CausalAttention`, `DecoderBlock`, `Transformer.forward`, `Transformer.generate`, `Transformer.get_loss_on_batch`, snapshot tests, and the TinyStories training loop.

The assignment-specific notes are in [notes.md](notes.md).

The implementation/debug companion is in [attention-sanity-checks.md](attention-sanity-checks.md).

## Checkpoint Plan

1. Attention exploration: soft lookup mechanics, copying, averaging, single-head variance, and multi-head intuition. Complete.
2. Position embeddings: permutation equivariance, why token order is invisible without position signals, and sinusoidal-position edge cases. Complete.
3. Transformer implementation: tensor-shape checklist for `model_solution.py`, training-loop notes, loss computation, generation, and local test commands. Complete.
4. Implementation sanity checks: focused debugging checklist for attention axes, causal masking, residual order, language-model loss, generation, and training sanity checks. Complete.

## Intended Local Structure

- `code/` for starter files once pulled locally
- `notes.md` for derivations and tensor-shape checks
- `attention-sanity-checks.md` for implementation pitfalls and debugging checks
