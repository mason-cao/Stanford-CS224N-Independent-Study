# A3 Training And Writeup Notes

## Why This File Exists

The implementation notes tell me how to make the Transformer pass the local tests. This file is for the last part of A3: training the small decoder-only model, reading the loss and gradient-norm curves, and writing up the assignment without treating the plot as a black box.

The important transition is:

```text
passing module tests -> correct next-token loss -> meaningful training curve
```

A decreasing training curve is only evidence after the model is already mechanically correct. If attention is leaking future tokens, if the loss is shifted incorrectly, or if generation is using the wrong logits, the curve can be misleading.

## What The Training Section Is Testing

The training part is not asking for a production-scale language model. It is checking whether I can connect the pieces from `model_solution.py` to a real optimization loop.

The assignment expects:

- a working `Transformer.get_loss_on_batch`
- a run of `train.py` for 100 batches
- a saved plot of training loss and gradient norm
- a short interpretation of whether the loss decreases
- optional bonus experiments that lower final loss after the same number of steps

The small model is intentionally underpowered. That is fine. The point is to confirm that the implementation can learn from sequence data at all.

## Data Flow Through Training

The training script uses a simple language-modeling setup:

```text
TinyStories text
-> GPT-2 tokenizer
-> token ID stream
-> fixed-length chunks
-> batches of token IDs
-> Transformer.get_loss_on_batch
-> optimizer step
```

The tokenized batch has shape:

```text
B x T
```

where `T` is usually the configured context length. Each row is a chunk of contiguous token IDs. The model does not receive a separate label tensor because next-token prediction creates labels by shifting the same sequence.

Inside the loss:

```text
input positions: x_0, x_1, ..., x_{T-2}
target tokens:   x_1, x_2, ..., x_{T-1}
```

The final token in the chunk is a target but not an input position with its own target inside that same chunk.

This is why the loss uses:

```text
logits[:, :-1, :]
input_ids[:, 1:]
```

If I accidentally use:

```text
logits[:, 1:, :]
input_ids[:, :-1]
```

then I am training the model to predict previous tokens from later positions. That is the wrong direction for an autoregressive decoder.

## Random Baseline Scale

For a vocabulary of size `V`, a uniform random predictor has cross-entropy:

```text
log(V)
```

The GPT-2 tokenizer vocabulary size is:

```text
V = 50257
```

So the uniform scale is:

```text
log(50257) ~= 10.82
```

This number is only a rough anchor. A randomly initialized Transformer is not guaranteed to produce exactly uniform probabilities, and the token distribution in TinyStories is not uniform. Still, the first loss should be in the same broad range as a difficult 50k-class prediction problem, not near zero.

If I see an initial loss close to zero, that is suspicious. Possible explanations:

- the labels are accidentally equal to something already visible
- the loss is averaged over the wrong tensor
- the batch contains degenerate data
- the code is not computing vocabulary cross-entropy

If I see a huge loss that does not move, I should check for exploding logits, a bad learning rate, or a broken target shift.

## Reading The Loss Curve

The expected baseline plot should show training loss over 100 batches. I should read it as an optimization diagnostic, not as a final model-quality score.

Useful interpretations:

- Downward trend: the model is learning local token patterns from the training chunks.
- No trend: the optimizer is not making useful progress, or the implementation is wrong.
- Sharp spikes: learning rate may be too high, gradients may be unstable, or batches may vary a lot.
- Smooth but slow decrease: implementation is probably sane, but the model or learning rate may be conservative.
- Very fast drop followed by flattening: the model may be memorizing easy local structure and then running into capacity or optimization limits.

The curve is over only 100 batches, so I should not overstate it. It shows that the implementation trains, not that the model is good.

A concise writeup sentence should sound like:

```text
The loss decreases over the 100 training batches, which suggests the Transformer implementation and next-token loss are wired correctly. Because this is a tiny model trained briefly on a small slice of TinyStories, the curve is a sanity check rather than a meaningful language-modeling benchmark.
```

## Reading The Gradient-Norm Curve

The gradient norm answers a different question from the loss:

```text
how large is the update signal before the optimizer step?
```

The norm should usually be nonzero. If it is exactly zero across training, I should suspect that:

- loss is detached from the model parameters
- `optimizer.zero_grad`, `loss.backward`, or `optimizer.step` is missing or misplaced
- all parameters have `requires_grad = False`
- the batch path is bypassing the model

If the gradient norm is enormous or repeatedly spikes, I should suspect:

- learning rate too high
- unstable logits
- missing gradient clipping
- numerical issue in attention masking
- data batch with unusually hard or long-range structure

If gradient clipping is enabled, the plotted norm may show the pre-clipping norm or post-clipping norm depending on how the script records it. That distinction matters. A post-clipping curve hides how large the raw gradients were.

The loss curve tells me whether the objective is improving. The gradient-norm curve tells me whether the optimization signal is stable enough to trust.

## AdamW Notes

The training script uses AdamW. I should remember the distinction between Adam and AdamW:

```text
Adam:  adaptive moments plus weight decay often implemented as L2-style coupling
AdamW: adaptive moments plus decoupled weight decay
```

The practical reason AdamW is standard for Transformers is that adaptive per-parameter scaling and decoupled weight decay behave better than plain SGD for this kind of model.

At a high level:

- the first moment tracks a smoothed gradient direction
- the second moment tracks a smoothed squared-gradient scale
- parameter updates are normalized by the second-moment estimate
- decoupled weight decay shrinks weights separately from the gradient update

This matters for A3 because a tiny Transformer has many matrices that can receive gradients at different scales: embeddings, attention projections, MLP layers, layer norms, and the output projection. Adaptive scaling makes the short training run much less fragile.

## Causal Mask Training Check

The causal mask is essential during training, not only during generation.

For a sequence:

```text
x_0, x_1, x_2, x_3
```

the model at position `1` is trained to predict `x_2`. It may attend to:

```text
x_0, x_1
```

It must not attend to:

```text
x_2, x_3
```

If future positions are visible during training, the loss can decrease for the wrong reason. The model could learn to copy the answer from the future instead of learning a valid left-to-right language model.

This is the reason a good training curve does not replace the attention unit test. The mask has to be right before the curve is meaningful.

## Why Greedy Generation Is The Tested Default

The assignment asks for greedy decoding in `generate`.

The loop is:

```text
condition on current tokens
compute logits
take last-position logits
choose argmax token
append token
repeat
```

Greedy generation is deterministic. That makes it appropriate for snapshot testing.

Sampling would require extra choices:

- temperature
- top-k or nucleus filtering
- random seed
- whether to sample from logits or probabilities

Those are useful for open-ended generation, but they add noise to an assignment whose purpose is to verify the model wiring.

## Baseline Writeup Checklist

For the required training plot, I should include:

- the loss and gradient-norm plot saved by the script
- whether the loss decreases over 100 batches
- whether the gradient norm looks stable, spiky, or suspicious
- a reminder that this is a tiny model and short run
- any implementation detail that affected training, such as gradient clipping

The answer should not overclaim. A good final paragraph:

```text
The training loss trends downward during the 100-batch run, and the gradient norm remains nonzero throughout training. This is the expected sanity-check behavior for a correctly wired next-token objective. The model is still very small and trained briefly, so I interpret the plot as evidence that the implementation learns, not as evidence of strong generation quality.
```

## Bonus Experiment Discipline

The bonus asks for three distinct changes that improve final loss after 100 steps. It explicitly does not count three values of the same idea as three ideas.

Good experiment categories:

- learning rate
- optimizer choice or optimizer parameters
- model width
- model depth
- number of heads
- batch size
- gradient clipping
- learning-rate schedule
- dropout or regularization
- weight initialization

The right process is:

```text
baseline
-> change one idea
-> record final loss
-> keep the improvement if it helps
-> add a second different idea
-> record final loss
-> add a third different idea
-> record final loss
```

This makes the improvements compound and keeps the explanation honest.

## Bonus Ideas I Would Try First

### 1. Learning Rate

The first experiment should be learning rate because it is cheap and often dominates short training runs.

Expected pattern:

- too low: loss decreases slowly or barely moves
- too high: loss spikes or becomes unstable
- reasonable: loss decreases faster without repeated spikes

This counts as one idea no matter how many values I try.

### 2. Model Width

Increasing `d_model` gives the model more capacity per token. It also changes the attention head dimension if the number of heads stays fixed.

The constraint remains:

```text
d_model % n_heads == 0
```

Potential benefit:

- richer token representations
- larger MLP hidden layer because it expands to `4 * d_model`

Potential cost:

- slower steps
- more parameters
- more risk of overfitting a tiny slice if training is extended

For the bonus, speed per wall-clock second matters less than final loss after a fixed 100 steps, unless the run becomes too slow to complete.

### 3. Learning-Rate Schedule Or Warmup

Transformers often benefit from a gentler early optimization phase. A simple schedule can help avoid unstable first steps while still allowing a useful learning rate later.

Expected benefit:

- fewer early gradient spikes
- smoother loss decrease
- better final loss if the baseline learning rate is aggressive

This is a different idea from choosing a single learning rate because the update scale changes over time.

## Bonus Results Table Template

I should record results in a table like this:

```text
Run | Change from previous run | Final loss after 100 steps | Notes
Baseline | none | ... | ...
1 | learning rate changed from ... to ... | ... | ...
2 | kept run 1 and changed ... | ... | ...
3 | kept run 2 and changed ... | ... | ...
```

If a change makes the final loss worse, it should not be counted as one of the bonus improvements. It can still be mentioned briefly as a failed experiment if useful.

## What I Need To Submit For A3

The handout expects the code submission to include the implemented files from the starter archive, packaged by the provided submission script. For the writeup, the relevant A3 pieces are:

- attention exploration answers
- position embedding proof and explanation
- training loss and gradient-norm plot
- brief interpretation of the training plot
- optional bonus changes, curves, and final losses

For this repository, I am not mirroring the starter code archive. The local study record should preserve:

- the derivations and implementation shape notes in `notes.md`
- the implementation debugging checklist in `attention-sanity-checks.md`
- the training and writeup interpretation notes in this file

## Final Mental Model

A3 starts with attention as a mathematical operation:

```text
softmax(query-key scores) applied to values
```

It then shows why position embeddings are necessary:

```text
without position signals, self-attention plus per-position MLPs are permutation equivariant
```

Finally, it turns those ideas into a decoder-only language model:

```text
token IDs
-> token embeddings + position embeddings
-> causal self-attention and MLP blocks
-> next-token logits
-> shifted cross-entropy loss
-> greedy autoregressive generation
```

The training plot is the practical closing loop. If the loss decreases while the gradient norm remains plausible, the pieces are no longer just individually correct; they are connected well enough for optimization to improve the model.
