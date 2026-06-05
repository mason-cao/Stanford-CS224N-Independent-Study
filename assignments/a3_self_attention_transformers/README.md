# A3: Self-Attention and Transformers

## Official Slot

- Course week: Week 3
- Official handout: <https://web.stanford.edu/class/cs224n/assignments_w26/a3.pdf>
- Official code release: <https://web.stanford.edu/class/cs224n/assignments_w26/a3.zip>

## Repo Status

Complete as a study-notes milestone.

I have completed the A3 study pass through the written material and starter-code structure: attention exploration, position embeddings, the decoder-only Transformer implementation map, implementation sanity checks, and training/writeup interpretation.

The first checkpoint is focused on attention as a soft lookup operation: how queries score keys, how softmax converts scores into weights, why copying a value is easy when one score dominates, why averaging values with one head is fragile, and why multiple heads give a cleaner way to compose information.

The second checkpoint is focused on why position information is mathematically necessary. Without position embeddings, the simplified Transformer block in the handout is permutation equivariant: if the input tokens are shuffled, the output is shuffled in exactly the same way. Position embeddings break that symmetry by making identical token embeddings different at different sequence positions.

The third checkpoint is focused on the implementation path through the official starter code: `MLP`, `CausalAttention`, `DecoderBlock`, `Transformer.forward`, `Transformer.generate`, `Transformer.get_loss_on_batch`, snapshot tests, and the TinyStories training loop.

The assignment-specific notes are in [notes.md](notes.md).

The implementation/debug companion is in [attention-sanity-checks.md](attention-sanity-checks.md).

The training and writeup companion is in [training-and-writeup.md](training-and-writeup.md).

## Wrap-Up

A3 is the point where the course material turns into the architecture behind modern language models. My main takeaways are:

- Attention is a differentiable lookup: query-key scores become a softmax distribution over values.
- One attention head can copy or average values, but multiple heads give a cleaner way to compose several focused lookups.
- A self-attention block without position information is permutation equivariant, so position embeddings are necessary for ordered text.
- Decoder-only language modeling needs both position embeddings and a causal mask: positions identify order, while the mask prevents future-token leakage.
- The implementation is mostly a shape-preserving residual stack from `B x T x d_model` to `B x T x d_model`, followed by a projection to vocabulary logits.
- The training curve only becomes meaningful after the attention mask, residual order, loss shift, and generation path are mechanically correct.

I have not mirrored the official starter archive into this repository. The repo version of A3 is complete as a study record: derivations, shape checks, debugging checklists, training interpretation notes, and the local implementation plan I would use with the official starter files.

## Checkpoint Plan

1. Attention exploration: soft lookup mechanics, copying, averaging, single-head variance, and multi-head intuition. Complete.
2. Position embeddings: permutation equivariance, why token order is invisible without position signals, and sinusoidal-position edge cases. Complete.
3. Transformer implementation: tensor-shape checklist for `model_solution.py`, training-loop notes, loss computation, generation, and local test commands. Complete.
4. Implementation sanity checks: focused debugging checklist for attention axes, causal masking, residual order, language-model loss, generation, and training sanity checks. Complete.
5. Training and writeup closure: TinyStories training flow, loss-curve interpretation, gradient-norm interpretation, baseline scale, and bonus experiment discipline. Complete.

## Intended Local Structure

- `code/` for starter files once pulled locally
- `notes.md` for derivations and tensor-shape checks
- `attention-sanity-checks.md` for implementation pitfalls and debugging checks
- `training-and-writeup.md` for training-curve interpretation and final writeup notes
