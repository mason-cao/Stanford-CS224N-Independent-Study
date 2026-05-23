# A3: Self-Attention and Transformers

## Official Slot

- Course week: Week 3
- Official handout: <https://web.stanford.edu/class/cs224n/assignments_w26/a3.pdf>
- Official code release: <https://web.stanford.edu/class/cs224n/assignments_w26/a3.zip>

## Repo Status

Started.

I have started the A3 materials by reading the official handout and Week 3 Transformer materials, then writing a first pass through the attention-exploration section.

The first checkpoint is focused on attention as a soft lookup operation: how queries score keys, how softmax converts scores into weights, why copying a value is easy when one score dominates, why averaging values with one head is fragile, and why multiple heads give a cleaner way to compose information.

The assignment-specific notes are in [notes.md](notes.md).

## Checkpoint Plan

1. Attention exploration: soft lookup mechanics, copying, averaging, single-head variance, and multi-head intuition.
2. Position embeddings: permutation equivariance, why token order is invisible without position signals, and sinusoidal-position edge cases.
3. Transformer implementation: tensor-shape checklist for `model_solution.py`, training-loop notes, loss computation, generation, and local test commands.

## Intended Local Structure

- `code/` for starter files once pulled locally
- `notes.md` for derivations and tensor-shape checks
- `attention-sanity-checks.md` for implementation pitfalls
