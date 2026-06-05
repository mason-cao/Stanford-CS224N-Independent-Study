# A3 Attention And Transformer Sanity Checks

## Purpose

These are the checks I want beside me when implementing the official A3 starter code. The main notes explain the math and architecture; this file is the practical debugging companion for `model_solution.py`, `train.py`, and the snapshot tests.

The key idea is that a decoder-only Transformer is a stack of shape-preserving updates. Once token IDs become embeddings, the model repeatedly transforms:

```text
B x T x d_model
```

until the final projection to vocabulary logits:

```text
B x T x vocab_size
```

Most bugs in this assignment are not deep conceptual mistakes. They are usually one of:

- wrong tensor axis
- wrong triangular mask direction
- wrong residual/layer-norm order
- wrong loss shift
- generation using the wrong time step
- accidentally changing a deterministic test into a stochastic one

## Shape Legend

I will use:

```text
B = batch size
T = sequence length for the current batch
L = context_length, the maximum sequence length configured for the model
C = d_model, the hidden width
H = n_heads
D = d_attention = C / H
V = vocab_size
```

For the snapshot-test model:

```text
C = 48
H = 2
D = 24
L = 16
V = 16
```

The attention split only works because:

```text
C % H == 0
```

If this is false, there is no clean way to divide the hidden dimension evenly across heads.

## Dependency Order

The implementation should move in this order:

```text
MLP.forward
-> CausalAttention.forward
-> DecoderBlock.forward
-> Transformer.forward
-> Transformer.get_loss_on_batch
-> Transformer.generate
-> train.py run
```

That is the same dependency graph as the model:

- `DecoderBlock` uses `MLP` and `CausalAttention`.
- `Transformer.forward` uses `DecoderBlock`.
- loss and generation both depend on `Transformer.forward`.
- training depends on the loss.

If an early module test fails, later test failures are usually not informative yet.

## MLP.forward

The MLP is the per-position feed-forward network inside each decoder block:

```text
Linear(C, 4C)
GELU
Linear(4C, C)
```

Input and output:

```text
x:      B x T x C
output: B x T x C
```

The MLP should not mix time positions. It uses the same feed-forward transformation at every row of the sequence. That is why the shape has the same `B` and `T` on both sides.

Sanity checks:

- If input has shape `B x T x C`, `fc1(x)` should have shape `B x T x 4C`.
- The activation should keep `B x T x 4C`.
- `fc2(...)` should return `B x T x C`.
- Use the exact GELU module supplied in the starter if snapshot tests expect it.

Conceptual check:

```text
attention mixes tokens across positions
MLP transforms each position independently
```

If I think the MLP needs the causal mask, I am mixing up the roles of the two sublayers.

## CausalAttention.forward: Full Tensor Path

Start with hidden states:

```text
x: B x T x C
```

The projection layers create keys, queries, and values:

```text
k = W_k(x): B x T x C
q = W_q(x): B x T x C
v = W_v(x): B x T x C
```

Split the hidden dimension into heads:

```text
B x T x C
-> B x T x H x D
-> B x H x T x D
```

The second form is the useful one for attention because each head should compute its own `T x T` score matrix.

Scores:

```text
scores = q @ k.transpose(-2, -1)
scores: B x H x T x T
```

The last two axes are:

```text
row    = query position
column = key/value position
```

Then scale:

```text
scores = scores / sqrt(D)
```

Use `D`, not `C`. Each head computes dot products in the `D`-dimensional head space, so the variance correction is based on `d_attention`.

Apply the causal mask before softmax:

```text
masked_scores: B x H x T x T
attention_weights = softmax(masked_scores, dim=-1)
```

The softmax dimension is the key-position axis. Every query position receives a probability distribution over the positions it is allowed to read.

Then apply values:

```text
head_output = attention_weights @ v
head_output: B x H x T x D
```

Concatenate heads:

```text
B x H x T x D
-> B x T x H x D
-> B x T x C
```

Then output projection:

```text
W_o(...): B x T x C
```

Final shape:

```text
CausalAttention(x): B x T x C
```

## Causal Mask Direction

For a sequence length of 4, a decoder token at position `t` may attend only to positions `0` through `t`.

Allowed positions:

```text
query row 0: key columns 0
query row 1: key columns 0, 1
query row 2: key columns 0, 1, 2
query row 3: key columns 0, 1, 2, 3
```

The allowed mask is lower triangular:

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

Future entries are above the diagonal. Those entries should be replaced with a very negative value before softmax:

```text
0 -inf -inf -inf
0    0 -inf -inf
0    0    0 -inf
0    0    0    0
```

After softmax, each row should sum to 1 over the allowed columns, and future columns should receive probability 0.

The mask in the module is usually registered at full context length:

```text
1 x 1 x L x L
```

For a current input length `T`, slice it to:

```text
1 x 1 x T x T
```

This matters because the model should handle shorter batches during testing and generation.

## DecoderBlock.forward

The A3 model uses pre-layer-norm residual blocks:

```text
x = x + attention(pre_layer_norm(x))
x = x + mlp(post_layer_norm(x))
```

This is not the same as:

```text
x = layer_norm(x + attention(x))
```

Both forms have compatible shapes, so a wrong order can run while still failing the snapshot tests.

Shape invariant:

```text
input:  B x T x C
output: B x T x C
```

The residual additions are only valid because attention and MLP both return `B x T x C`.

Conceptual check:

- The residual path preserves the current representation.
- The normalized sublayer computes an update.
- The block adds that update back to the residual stream.

## Transformer.forward

The model input is token IDs:

```text
input_ids: B x T
```

Token embeddings:

```text
token_embeddings(input_ids): B x T x C
```

Position IDs:

```text
positions = [0, 1, ..., T - 1]
```

Position embeddings:

```text
position_embeddings(positions): T x C
```

When added to token embeddings, `T x C` broadcasts across the batch:

```text
h = token_emb + pos_emb
h: B x T x C
```

Then:

```text
for block in backbone:
    h = block(h)

h = final_layer_norm(h)
logits = lm_head(h)
```

Final output:

```text
logits: B x T x V
```

The row `logits[b, t, :]` is the score vector for the next token after the model has read positions up through `t`.

Forward-pass checks:

- `T` must not exceed `context_length`.
- Position IDs should be created on the same device as `input_ids`.
- Position embeddings must use the current `T`, not the maximum `L`, unless they are sliced.
- The final layer norm happens before the vocabulary projection.

## Transformer.get_loss_on_batch

For language modeling, one sequence supplies both context and labels.

Example:

```text
input_ids = [x0, x1, x2, x3, x4]
```

The training targets are:

```text
predict x1 from position 0
predict x2 from position 1
predict x3 from position 2
predict x4 from position 3
```

The last position has no next-token target inside the same chunk.

So:

```text
logits: B x T x V
labels: B x T
```

Shift:

```text
prediction logits = logits[:, :-1, :]
target labels     = labels[:, 1:]
```

Flatten for cross-entropy:

```text
prediction logits: B x (T - 1) x V -> B*(T - 1) x V
target labels:     B x (T - 1)     -> B*(T - 1)
```

Cross-entropy should receive raw logits, not probabilities. It applies log-softmax internally.

Sign check:

- If the correct next token receives a high logit, the loss should be lower.
- If the correct next token receives a low logit, the loss should be higher.

This is the same `yhat - y` classification idea from A2, repeated at every next-token position.

## Transformer.generate

Generation uses the trained model autoregressively:

```text
repeat num_new_tokens times:
    crop context to the last L tokens
    compute logits
    take the logits at the final time step
    choose the next token
    append the next token
```

The final time step matters:

```text
last_logits = logits[:, -1, :]
```

Using logits from all positions would produce too many tokens per step. Using position `0` would ignore the current end of the prompt.

The A3 handout specifies greedy decoding for the tested implementation:

```text
next_token = argmax(last_logits)
```

Sampling, temperature, and top-k filtering are useful generation tools, but they are the wrong default here because the snapshot test expects deterministic output.

Context cropping:

```text
idx_cond = idx[:, -L:]
```

This prevents the model from receiving sequences longer than its learned position table and causal mask.

## Snapshot Test Debugging

Snapshot tests are useful because they make numerically silent bugs visible. A wrong implementation can preserve every shape and still fail the saved outputs.

When a snapshot test fails, I should ask these questions in order:

1. Did the module preserve the expected input/output shape?
2. Did I use the exact activation and layer order from the starter?
3. Did I transpose only the intended dimensions?
4. Did I apply softmax over the key-position axis?
5. Did I mask future tokens before softmax?
6. Did I use `sqrt(D)` instead of `sqrt(C)`?
7. Did I keep generation greedy and deterministic?
8. Did I shift logits and labels correctly for loss?

The expected local workflow from the starter archive is:

```bash
cd tests
pytest test_student.py::test_mlp
pytest test_student.py::test_attention
pytest test_student.py::test_decoder_block
pytest test_student.py::test_forward
pytest test_student.py::test_loss_on_batch
pytest test_student.py::test_generate
```

Then run the full suite:

```bash
pytest
```

## Training Sanity Checks

Training should come after tests. If the tests fail, a training curve is not trustworthy.

For a correct tiny model, I should expect:

- loss starts near the random-prediction range
- loss trends downward over 100 batches
- gradient norm is nonzero during learning
- very large gradient spikes may indicate instability or a learning rate issue
- a flat loss curve after passing tests suggests data, optimizer, learning rate, or mode-setting issues

The rough random baseline for next-token prediction over the GPT-2 vocabulary is:

```text
log(50257) ~= 10.82
```

That is not an exact expected starting loss because initialization, token distribution, batch size, and sequence packing all matter. It is a useful scale check. A fresh model predicting almost uniformly over 50k tokens should be around that order of magnitude, not near zero.

## Bonus Speedup Ideas

The bonus asks for distinct types of changes, not three different values for the same knob. Useful categories to test separately:

- learning-rate tuning
- optimizer changes
- batch-size or gradient-accumulation changes
- model-width/depth changes
- dropout or regularization changes
- learning-rate scheduling
- weight initialization adjustments

For each idea, the evidence should include:

- what changed
- what stayed fixed
- final loss after 100 steps
- learning curve against the baseline
- whether the improvement compounded with previous changes

The clean experiment rule is to change one idea at a time before combining improvements. Otherwise I cannot tell which change caused the lower loss.

## Minimal Mistake Log Template

When a test fails, write down:

```text
failed test:
symptom:
expected shape:
actual shape:
suspected axis/order issue:
fix tried:
result:
```

This keeps debugging from turning into random edits. A3 is small enough that each failure should usually point to a specific axis, mask, or layer-order mistake.

## Final A3 Implementation Checklist

- `MLP.forward` preserves `B x T x C`.
- Attention splits into `B x H x T x D`.
- Attention scores have shape `B x H x T x T`.
- Scores are scaled by `sqrt(D)`.
- Future positions are masked before softmax.
- Attention softmax runs over key positions.
- Heads are concatenated back to `B x T x C`.
- Decoder block uses pre-norm residual order.
- `Transformer.forward` adds token and position embeddings before blocks.
- Final layer norm runs before `lm_head`.
- Loss uses `logits[:, :-1, :]` and `input_ids[:, 1:]`.
- Cross-entropy receives flattened raw logits and flattened integer labels.
- Generation crops to `context_length`.
- Generation uses final-position logits.
- Generation uses greedy `argmax` for the tested version.
- Unit tests pass before training.
- Training produces a decreasing loss curve before any bonus experiments.
