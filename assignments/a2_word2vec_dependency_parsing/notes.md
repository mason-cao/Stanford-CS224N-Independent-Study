# A2 Notes: Word2Vec And Dependency Parsing

## Sources

- Official A2 handout: <https://web.stanford.edu/class/cs224n/assignments_w26/a2.pdf>
- Official A2 code release: <https://web.stanford.edu/class/cs224n/assignments_w26/a2.zip>
- Week 2 neural network notes: <../../notes/lecture-notes/02_neural_networks_backprop.md>

## What This Assignment Is Testing

A2 is doing two things at once. Part 1 checks whether I can derive gradients for Word2Vec without hiding behind a framework. Part 3 then asks me to implement a small neural dependency parser in PyTorch, where the framework handles gradients but I still need to understand the tensor shapes and training loop.

The assignment flow is:

- Word2Vec math: naive softmax, cross-entropy, center vectors, outside vectors, and activation derivatives.
- Optimization concepts: Adam as momentum plus adaptive per-parameter scaling, and dropout as regularization.
- Parser mechanics: stack, buffer, dependencies, transitions, minibatch parsing.
- Parser model: embedding lookup, hidden layer, logits, cross-entropy training, dev/test UAS.

I should not treat these as disconnected sections. The dependency parser is just the Week 2 feedforward classifier applied to parser states. The parser state gets featurized into token indices; token indices become embeddings; concatenated embeddings become the input vector; the model predicts one of three transitions.

## Part 1: Word2Vec Derivation Map

The handout uses a column-vector convention:

```text
u_w in R^d
v_c in R^d
U in R^{d x |V|}
```

The columns of `U` are outside word vectors. The predicted scores for all outside words are:

```text
z = U^T v_c
```

So:

```text
z in R^{|V|}
yhat = softmax(z)
y in R^{|V|}
```

The loss for one center/outside pair is:

```text
J = -log yhat_o
```

The useful backprop view:

```text
v_c -> z = U^T v_c -> yhat = softmax(z) -> J = CE(y, yhat)
```

The score-layer error signal is:

```text
dJ/dz = yhat - y
```

From there, the remaining gradients are shape bookkeeping. Since `z = U^T v_c`, the gradient with respect to `v_c` must have shape `R^d`, and the gradient with respect to each outside vector `u_w` must also have shape `R^d`.

The mistake I am most likely to make is copying a formula from notes with the opposite orientation. I need to check dimensions before trusting a transpose.

## Cross-Entropy With A One-Hot Label

For one training example, the empirical distribution `y` is one-hot. If the true outside word is `o`, then:

```text
y_o = 1
y_w = 0 for w != o
```

So the full cross-entropy:

```text
-sum_w y_w log(yhat_w)
```

collapses to the single term for the true outside word:

```text
-log(yhat_o)
```

This is not a special Word2Vec trick. It is the standard multiclass classification loss when the target is one class.

## Interpreting The Center-Vector Gradient

The gradient for the center vector compares two things:

- what the model expected under its current distribution
- what the data said should happen for this example

During gradient descent, subtracting the gradient should increase compatibility with the true outside word and decrease compatibility with outside words that received too much probability.

The conceptual split:

- the `yhat` part is the model's current weighted average of outside vectors
- the `y` part isolates the observed outside vector

If the predicted distribution exactly matched the target distribution for the example, the softmax error signal would be zero. In practice, exact equality is an idealized condition because softmax probabilities are rarely exactly one-hot with finite scores.

## Gradients For Outside Vectors

Each outside vector gets an update proportional to the center vector. The scalar multiplier is the prediction error for that outside word.

For the true outside word, the multiplier should be negative before the gradient descent subtraction, so subtracting the gradient pulls that outside vector toward the center vector.

For an incorrect outside word, the multiplier is positive when the model assigned it nonzero probability, so subtracting the gradient pushes it away from the center vector. The amount depends on how much probability the model gave that wrong word.

This makes the training pressure contrastive even in naive softmax. The model is not only making one pair closer; it is redistributing probability across the whole vocabulary.

## L2 Normalization Question

L2 normalization removes vector magnitude and keeps direction:

```text
u_normalized = u / ||u||
```

That can be useful when direction carries the semantic signal I care about. It can be harmful when magnitude carries information that the downstream classifier should use.

The handout's hint about `u_x = alpha u_y` is the key. If `alpha > 0`, the normalized vectors point in the same direction even if their original lengths differ. So normalization collapses distinctions that are only encoded in length. Whether that matters depends on whether length was meaningful for the downstream task.

## Activation Derivatives

For leaky ReLU:

```text
f(x) = max(alpha x, x)
```

The derivative is piecewise:

```text
x > 0: slope 1
x < 0: slope alpha
```

For sigmoid:

```text
sigma(x) = 1 / (1 + exp(-x))
```

The derivative can be expressed in terms of the output itself:

```text
sigma'(x) = sigma(x) * (1 - sigma(x))
```

That form is useful in backprop because the forward pass already computed `sigma(x)`.

## Part 2: Adam Notes

Plain stochastic gradient descent updates parameters directly in the direction of the current minibatch gradient:

```text
theta_{t+1} = theta_t - alpha * grad_t
```

Adam starts by smoothing the gradient with a moving average:

```text
m_{t+1} = beta1 * m_t + (1 - beta1) * grad_t
```

This is momentum. It reduces update noise by mixing the current minibatch gradient with recent gradient history. If several minibatches agree on a direction, the update builds confidence in that direction. If gradients bounce back and forth, the moving average dampens the oscillation.

Adam also tracks a moving average of squared gradients:

```text
v_{t+1} = beta2 * v_t + (1 - beta2) * (grad_t * grad_t)
```

Then it divides by a square-rooted version of this second moment. Parameters with consistently large gradients get scaled down; parameters with smaller historical gradients can receive relatively larger effective steps. The point is not that Adam knows the perfect learning rate. It makes the update less sensitive to uneven gradient scales across parameters.

## Part 2: Dropout Notes

Dropout randomly zeros hidden units during training:

```text
h_drop = gamma * d * h
```

where each mask entry is:

```text
d_i = 0 with probability p_drop
d_i = 1 with probability 1 - p_drop
```

The scaling factor `gamma` is chosen so the expected dropped-out activation equals the original activation:

```text
E[h_drop_i] = h_i
```

That means the surviving units are scaled upward during training. In PyTorch, `nn.Dropout` handles this inverted-dropout scaling automatically.

Why it helps:

- the model cannot rely on one hidden unit always being present
- hidden units are pressured to learn less brittle features
- it acts like training an ensemble of thinned networks that share parameters

Why it should be off during evaluation:

- evaluation should be deterministic
- all learned units should be available
- the train-time scaling already accounts for the expected activation magnitude

In code, this is why `model.train()` and `model.eval()` matter.

## Part 3: Transition-Based Parser Mechanics

The parser state has three pieces:

- stack: words currently being processed
- buffer: words not yet shifted
- dependencies: predicted head/dependent pairs

Initial state:

```text
stack = [ROOT]
buffer = sentence words in order
dependencies = []
```

Transitions:

```text
S:  move the first buffer item to the top of the stack
LA: top stack item becomes head of second stack item; remove second stack item
RA: second stack item becomes head of top stack item; remove top stack item
```

The complete parse ends when:

```text
buffer is empty
stack has one item
```

For a sentence with `n` words, the parser needs `n` SHIFT operations to move all words out of the buffer, plus `n` arc operations to attach all words into the tree. So the full parse should take `2n` transitions when ROOT is initialized separately on the stack.

## Starter Code Map

The A2 code archive has these files:

- `parser_transitions.py`: implement `PartialParse.__init__`, `parse_step`, and `minibatch_parse`.
- `parser_model.py`: implement parser parameters, embedding lookup, and forward pass.
- `run.py`: construct optimizer/loss, run forward pass, backpropagate, and step optimizer.
- `utils/parser_utils.py`: feature extraction, legality masks, data loading, minibatches, and UAS evaluation.
- `data/`: train/dev/test CoNLL files and `en-cw.txt` pretrained vectors.

I am not copying the starter archive into this repo yet. For now, this note is the plan I want in place before implementation.

## Parser Transition Implementation Plan

For `PartialParse.__init__`:

- copy the sentence into the buffer so the input list is not mutated
- initialize stack with the root token
- initialize dependencies as an empty list

For `parse_step`:

- `S`: pop from front of buffer, append to stack
- `LA`: add dependency `(stack[-1], stack[-2])`, then remove `stack[-2]`
- `RA`: add dependency `(stack[-2], stack[-1])`, then remove `stack[-1]`

The danger is removing the wrong stack element after adding an arc. The stack top is the last element of the list.

For `minibatch_parse`:

- create one `PartialParse` per sentence
- keep a shallow copied list of unfinished parses
- repeatedly take the first `batch_size` unfinished parses
- ask the model for transitions
- apply one transition to each parse in the minibatch
- filter out completed parses
- return dependencies in the original sentence order

The warning about not using `del` matters because the unfinished list shares object references with the full parse list.

## Parser Model Shape Notes

The handout defines:

```text
w = [w_1, ..., w_m]
E in R^{|V| x d}
x = [E_{w_1}, ..., E_{w_m}] in R^{d m}
h = ReLU(x W + b1)
l = h U + b2
yhat = softmax(l)
```

The starter code defaults:

```text
n_features = 36
hidden_size = 200
n_classes = 3
```

If embeddings have dimension `embed_size`, then:

```text
input_size = n_features * embed_size
embed_to_hidden_weight shape: input_size x hidden_size
embed_to_hidden_bias shape: hidden_size
hidden_to_logits_weight shape: hidden_size x n_classes
hidden_to_logits_bias shape: n_classes
```

The model returns logits, not softmax probabilities, because `CrossEntropyLoss` combines log-softmax and negative log likelihood internally.

For embedding lookup:

```text
w shape: batch_size x n_features
selected embeddings shape: batch_size x n_features x embed_size
flattened x shape: batch_size x (n_features * embed_size)
```

The implementation should use tensor indexing or gather-style operations, not `nn.Embedding`, because the assignment is testing the lookup operation directly.

## Training Loop Plan

Training loop for each minibatch:

```text
optimizer.zero_grad()
logits = model(train_x)
loss = loss_func(logits, train_y)
loss.backward()
optimizer.step()
```

Important details:

- `train_x` should be integer token indices.
- `train_y` should be class indices, not one-hot vectors, for `CrossEntropyLoss`.
- `model.train()` enables dropout.
- `model.eval()` disables dropout for dev/test parsing.
- Do not apply softmax before `CrossEntropyLoss`.

The handout gives expected rough targets:

- debug mode: loss below 0.2 and dev UAS above 65
- full training: train loss below 0.08 and dev UAS above 87

These are sanity targets, not proofs of correctness. Passing the small tests and hitting a reasonable UAS still leaves room for edge-case bugs in transition mechanics.

## Debugging Checklist

- Run `python parser_transitions.py part_c` after transition mechanics.
- Run `python parser_transitions.py part_d` after minibatch parsing.
- Run `python parser_model.py -e` after embedding lookup.
- Run `python parser_model.py -f` after forward pass.
- Run `python run.py -d` before full training.
- Only run full `python run.py` after debug mode is stable.

When something fails, first check shapes:

- Is the stack top at the end of the list?
- Did I mutate the original sentence?
- Are logits shaped `(batch_size, 3)`?
- Is `train_y` a 1D class-index tensor?
- Did I accidentally apply softmax before the loss?
- Is dropout only active in training mode?

## Questions I Need To Re-Derive By Hand

- Why the one-hot cross-entropy reduces to the negative log probability of the true outside word.
- How the `yhat - y` error signal follows from softmax plus cross-entropy.
- How the center-vector gradient changes shape under column-vector versus row-vector conventions.
- Why the outside-vector update is attractive for the true outside word and repulsive for overpredicted wrong words.
- Why the parser needs `2n` transitions for a sentence with `n` words.
- How the parser model dimensions line up from embedding lookup through logits.

## Current Status

I have read the handout and inspected the starter file structure. I have not pulled the code archive into this repo or started implementation.
