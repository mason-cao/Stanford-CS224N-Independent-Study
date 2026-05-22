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

## Part 1 Worked Derivation Notes

I want the Word2Vec part to feel like ordinary backpropagation, not like a bag of memorized formulas. The computational graph for one center/outside pair is:

```text
v_c -> z = U^T v_c -> yhat = softmax(z) -> J = -sum_w y_w log(yhat_w)
```

where:

```text
v_c in R^d
U in R^{d x |V|}
z in R^{|V|}
yhat in R^{|V|}
y in R^{|V|}
```

The handout's shape convention is the anchor: every gradient must have the same shape as the variable it differentiates.

### Cross-Entropy Collapse

Since `y` is one-hot for this single example, only the true outside word `o` contributes:

```text
-sum_w y_w log(yhat_w)
= -(1 * log(yhat_o) + sum_{w != o} 0 * log(yhat_w))
= -log(yhat_o)
```

This is why the naive-softmax Word2Vec loss is just the negative log probability assigned to the observed outside word.

### Softmax Plus Cross-Entropy Error Signal

For logits `z`, softmax gives:

```text
yhat_i = exp(z_i) / sum_j exp(z_j)
```

The derivative of the loss with respect to each logit has the standard form:

```text
dJ/dz_i = yhat_i - y_i
```

I can remember this as "predicted probability minus target probability." It also passes the sign test:

- for the true outside word, `y_i = 1`, so `yhat_i - y_i` is negative unless the model already assigns probability 1
- for wrong outside words, `y_i = 0`, so the gradient is positive in proportion to the probability the model assigned

Gradient descent subtracts this signal, so it increases the true logit and decreases overpredicted wrong logits.

### Center-Vector Gradient

The score equation is:

```text
z = U^T v_c
```

The upstream gradient is:

```text
dJ/dz = yhat - y
```

Since `U^T` maps `R^d` to `R^{|V|}`, the backward pass maps the error signal back through `U`:

```text
dJ/dv_c = U (yhat - y)
```

Shape check:

```text
U shape: d x |V|
yhat - y shape: |V|
result shape: d
```

Expanding the expression helps with intuition:

```text
U yhat - U y
```

The first term is the model's expected outside vector under its current distribution. The second term is the true outside vector, because multiplying `U` by one-hot `y` selects column `u_o`. So the gradient is:

```text
expected outside vector - observed outside vector
```

Subtracting the gradient moves `v_c` toward the observed outside vector and away from the expectation the model currently has wrong.

### When The Center Gradient Is Zero

The center-vector gradient is zero when:

```text
U (yhat - y) = 0
```

The strongest intuitive case is `yhat = y`, but that is not the only mathematical possibility if `yhat - y` lies in the null space of `U`. For the assignment answer, the compact equation above is the shape-correct condition.

### Outside-Vector Gradient

Each score is:

```text
z_w = u_w^T v_c
```

So each outside vector gets:

```text
dJ/du_w = (yhat_w - y_w) v_c
```

There are two cases:

```text
w = o:     dJ/du_w = (yhat_o - 1) v_c
w != o:   dJ/du_w = yhat_w v_c
```

The sign matters. For the true outside word, `yhat_o - 1` is negative, so subtracting the gradient adds a positive multiple of `v_c` to `u_o`. For a wrong outside word, `yhat_w` is positive, so subtracting the gradient moves `u_w` away from `v_c`.

The full matrix gradient with respect to `U` is the columns of those vector gradients:

```text
dJ/dU = [dJ/du_1, dJ/du_2, ..., dJ/du_|V|]
```

Equivalently:

```text
dJ/dU = v_c (yhat - y)^T
```

Shape check:

```text
v_c shape: d x 1
(yhat - y)^T shape: 1 x |V|
result shape: d x |V|
```

That matches `U`.

### L2 Normalization Answer Shape

If:

```text
u_x = alpha u_y
```

with `alpha > 0`, then:

```text
u_x / ||u_x|| = u_y / ||u_y||
```

So normalization collapses the two vectors onto the same direction. That is fine if the downstream task should treat them as equivalent except for scale. It loses useful information if the magnitude difference matters for classification.

In the handout's phrase-classification setup, if two words point in the same direction but have different strengths of positive or negative sentiment, raw magnitude could affect the sign or confidence of the summed phrase vector. Normalization would erase that intensity difference.

### Activation Derivative Answers

For leaky ReLU:

```text
f(x) = max(alpha x, x)
```

with `0 < alpha < 1`, the derivative is:

```text
x > 0: 1
x < 0: alpha
```

At `x = 0`, the derivative is not defined in the ordinary sense, and the handout says to ignore that case.

For sigmoid:

```text
sigma(x) = 1 / (1 + exp(-x))
```

The derivative is:

```text
sigma'(x) = sigma(x) (1 - sigma(x))
```

This form is the one to remember for backprop because the forward pass already computed `sigma(x)`.

### Part 1 Postmortem

The main thing I learned is that most of A2 Part 1 is one error signal reused through different variables:

```text
yhat - y
```

The rest is shape discipline. If I know whether vectors are columns or rows, I can usually recover the transpose rather than guessing. The center-vector update, outside-vector update, and parser classifier derivative later in the assignment are all the same softmax-cross-entropy pattern.

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

## Part 2 Worked Optimization Notes

This section is short in the handout, but it matters because the parser later uses these exact training ideas. Adam controls update behavior; dropout controls overfitting behavior.

### Adam: Momentum

Plain minibatch SGD uses only the current minibatch gradient:

```text
theta_{t+1} = theta_t - alpha * grad_t
```

The problem is that a minibatch gradient is noisy. One batch can point partly in a useful direction and partly in a direction caused by that batch's accidental sample composition.

Adam's first moving average is:

```text
m_{t+1} = beta1 * m_t + (1 - beta1) * grad_t
```

This is a low-pass filter on gradients. If consecutive minibatches point in a similar direction, the moving average preserves that direction. If a minibatch gives an unusual gradient, the average dampens it. That lower variance helps learning because the optimizer is less likely to zig-zag from batch noise and more likely to make steady progress along directions that persist across many batches.

I should not describe momentum as "avoiding local minima" in a vague way. The clean answer is about smoothing update directions and reducing variance.

### Adam: Adaptive Scaling

Adam's second moving average tracks squared gradients:

```text
v_{t+1} = beta2 * v_t + (1 - beta2) * (grad_t * grad_t)
```

The update divides by `sqrt(v)`:

```text
theta_{t+1} = theta_t - alpha * m_{t+1} / sqrt(v_{t+1})
```

Ignoring small numerical-stability constants, parameters with smaller historical squared gradients get larger effective updates, and parameters with larger historical squared gradients get smaller effective updates.

That helps when different parameters naturally have different gradient scales. In the parser, frequently observed embeddings or weights connected to common parser features may receive large gradients often, while rare-word features may receive smaller or less frequent gradients. Adaptive scaling keeps one parameter group from dominating only because its gradients are numerically larger.

The important caveat: Adam changes the step size, not the objective. It can make optimization easier, but it does not replace good features, correct labels, or a correct model implementation.

### Dropout Scaling

Dropout is defined as:

```text
h_drop = gamma * d * h
```

For one hidden unit:

```text
d_i = 0 with probability p_drop
d_i = 1 with probability 1 - p_drop
```

The assignment chooses `gamma` so:

```text
E[h_drop_i] = h_i
```

Compute the expectation:

```text
E[h_drop_i]
= E[gamma * d_i * h_i]
= gamma * h_i * E[d_i]
= gamma * h_i * (1 - p_drop)
```

Set this equal to `h_i`:

```text
gamma * h_i * (1 - p_drop) = h_i
gamma = 1 / (1 - p_drop)
```

This is inverted dropout. The units that survive during training are scaled up, so the expected activation magnitude matches the non-dropout network.

### Why Dropout Is Training-Only

Dropout should be active during training because it prevents the hidden layer from relying on a brittle exact set of active units. A feature has to be useful even when some neighboring units are absent. That pressure makes the learned representation less dependent on any single hidden unit.

Dropout should be off during evaluation because evaluation should measure the full learned model deterministically. If dropout stayed on, the same sentence could produce different parser decisions on different passes. Since the train-time scaling already preserved expected activation size, no random masking is needed at evaluation time.

### Part 2 Postmortem

The clean mental split:

- Adam changes how gradients become parameter updates.
- Dropout changes the training-time computation graph to regularize hidden representations.

Both are general neural network tools. A2 includes them before the parser because the parser is just a small neural classifier trained on parser-state features.

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

## Part 3 Worked Transition Notes

The handout sentence is:

```text
I presented my findings at the NLP conference
```

There are eight non-root tokens, so the parse should take:

```text
2n = 16 transitions
```

The first three transitions are given in the handout:

```text
[ROOT] -> SHIFT I
[ROOT, I] -> SHIFT presented
[ROOT, I, presented] -> LEFT-ARC presented -> I
```

Using the current handout's dependency tree, the transition sequence is:

| Step | Stack after step | Buffer after step | New dependency | Transition |
| --- | --- | --- | --- | --- |
| 0 | `[ROOT]` | `[I, presented, my, findings, at, the, NLP, conference]` |  | Initial |
| 1 | `[ROOT, I]` | `[presented, my, findings, at, the, NLP, conference]` |  | SHIFT |
| 2 | `[ROOT, I, presented]` | `[my, findings, at, the, NLP, conference]` |  | SHIFT |
| 3 | `[ROOT, presented]` | `[my, findings, at, the, NLP, conference]` | `presented -> I` | LEFT-ARC |
| 4 | `[ROOT, presented, my]` | `[findings, at, the, NLP, conference]` |  | SHIFT |
| 5 | `[ROOT, presented, my, findings]` | `[at, the, NLP, conference]` |  | SHIFT |
| 6 | `[ROOT, presented, findings]` | `[at, the, NLP, conference]` | `findings -> my` | LEFT-ARC |
| 7 | `[ROOT, presented]` | `[at, the, NLP, conference]` | `presented -> findings` | RIGHT-ARC |
| 8 | `[ROOT, presented, at]` | `[the, NLP, conference]` |  | SHIFT |
| 9 | `[ROOT, presented, at, the]` | `[NLP, conference]` |  | SHIFT |
| 10 | `[ROOT, presented, at, the, NLP]` | `[conference]` |  | SHIFT |
| 11 | `[ROOT, presented, at, the, NLP, conference]` | `[]` |  | SHIFT |
| 12 | `[ROOT, presented, at, the, conference]` | `[]` | `conference -> NLP` | LEFT-ARC |
| 13 | `[ROOT, presented, at, conference]` | `[]` | `conference -> the` | LEFT-ARC |
| 14 | `[ROOT, presented, conference]` | `[]` | `conference -> at` | LEFT-ARC |
| 15 | `[ROOT, presented]` | `[]` | `presented -> conference` | RIGHT-ARC |
| 16 | `[ROOT]` | `[]` | `ROOT -> presented` | RIGHT-ARC |

The stack discipline is the part that matters for implementation:

- `LEFT-ARC` removes the second item from the top of the stack.
- `RIGHT-ARC` removes the top item from the stack.
- `SHIFT` is the only operation that consumes the buffer.

The table also makes the `2n` result concrete. Each of the eight tokens is shifted once, and each of the eight tokens eventually gets attached by one arc operation.

### Why ROOT Gets An Arc

The sentence has eight real words, but the final stack includes `ROOT` and the main predicate `presented`. To finish, the parser still needs to attach the sentence head to ROOT:

```text
ROOT -> presented
```

That final `RIGHT-ARC` removes `presented`, leaving only `[ROOT]`. The stopping condition is therefore not merely "buffer is empty"; it is:

```text
buffer is empty and stack has exactly one item
```

### Implementation Implication

The transition table is also a unit test I can run mentally against `parse_step`:

```text
SHIFT:     buffer.pop(0) then stack.append(word)
LEFT-ARC:  dependencies.append((stack[-1], stack[-2])) then remove stack[-2]
RIGHT-ARC: dependencies.append((stack[-2], stack[-1])) then remove stack[-1]
```

If I remove the wrong item, the next row of the transition table becomes impossible. That is why I should debug transition mechanics before touching the neural model.

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

## Parser Model Worked Notes

The parser model is a feedforward classifier over parser states. The parser state itself is symbolic, but `utils/parser_utils.py` turns it into a fixed-length list of token indices. The model's job is to map those indices to one of three transition classes:

```text
SHIFT, LEFT-ARC, RIGHT-ARC
```

The assignment writes the model as:

```text
x = [E_{w_1}, ..., E_{w_m}]
h = ReLU(x W + b1)
l = h U + b2
yhat = softmax(l)
```

In implementation, the model should return `l`, the logits, not `yhat`. PyTorch's `CrossEntropyLoss` expects raw logits and applies log-softmax internally.

### Parameter Shapes

The starter defaults give:

```text
n_features = 36
hidden_size = 200
n_classes = 3
```

If:

```text
embed_size = d
batch_size = B
```

then:

```text
self.pretrained_embeddings shape: vocab_size x d
self.embed_to_hidden_weight shape: (36d) x 200
self.embed_to_hidden_bias shape: 200
self.hidden_to_logits_weight shape: 200 x 3
self.hidden_to_logits_bias shape: 3
```

The forward-pass shapes should be:

```text
w: B x 36
embedding_lookup(w): B x (36d)
h: B x 200
logits: B x 3
```

The easiest way to catch mistakes is to write the shape next to every matrix multiplication:

```text
(B x 36d) @ (36d x 200) + (200) -> B x 200
(B x 200) @ (200 x 3) + (3) -> B x 3
```

The bias vectors broadcast across the batch dimension.

### Manual Linear Layers

The handout explicitly says not to use `torch.nn.Linear` or `torch.nn.Embedding`. That means the layer operations should be written directly with parameters:

```python
x = self.embedding_lookup(w)
h = torch.relu(x @ self.embed_to_hidden_weight + self.embed_to_hidden_bias)
h = self.dropout(h)
logits = h @ self.hidden_to_logits_weight + self.hidden_to_logits_bias
return logits
```

The exact attribute names should match the starter TODOs, because the grader may inspect them. The point is not to avoid matrix multiplication; the point is to show that I know what a linear layer and an embedding lookup are doing.

### Efficient Embedding Lookup

The input `w` is a matrix of integer token ids:

```text
w shape: B x 36
```

The embedding matrix is:

```text
self.pretrained_embeddings shape: vocab_size x d
```

Indexing the embedding matrix by `w` should produce:

```text
self.pretrained_embeddings[w] shape: B x 36 x d
```

Then flatten the last two dimensions:

```text
B x 36 x d -> B x (36d)
```

A direct implementation pattern is:

```python
embeddings = self.pretrained_embeddings[w]
x = embeddings.view(w.shape[0], -1)
```

or, if contiguity becomes an issue:

```python
x = embeddings.reshape(w.shape[0], -1)
```

The important thing is that this avoids Python loops over examples or features. A nested loop would be easier to read but much slower and could fail grading time limits.

### ReLU Derivative For The Written Question

The parser written section asks for:

```text
h = ReLU(xW + b1)
```

Let:

```text
a = xW + b1
h_i = ReLU(a_i)
a_i = sum_k x_k W_{k,i} + b1_i
```

For one input coordinate `x_j` and one hidden unit `h_i`:

```text
dh_i/dx_j = 1[a_i > 0] * W_{j,i}
```

If the pre-activation `a_i` is positive, ReLU passes the local derivative through. If `a_i` is negative, the derivative is zero. The assignment says to ignore the undefined case at exactly zero.

### Softmax Classifier Derivative

The parser classifier has three logits:

```text
l in R^3
yhat = softmax(l)
J = CE(y, yhat)
```

For class index `i`:

```text
dJ/dl_i = yhat_i - y_i
```

If the true class is `c`, then:

```text
i = c:    dJ/dl_i = yhat_i - 1
i != c:  dJ/dl_i = yhat_i
```

This is the same result as the Word2Vec derivative. The only difference is that the parser has three transition classes instead of `|V|` possible outside words.

### Why The Model Is Small But Easy To Break

The architecture is simple, but several mistakes can produce plausible-looking code:

- returning probabilities instead of logits
- flattening embeddings in the wrong order
- using `nn.Embedding` or `nn.Linear` despite the assignment restriction
- accidentally detaching pretrained embeddings from the computation graph if they are supposed to train
- applying dropout during evaluation
- creating parameters as plain tensors instead of registered `nn.Parameter`s

The safest implementation order is:

1. parameter shapes
2. embedding lookup sanity check
3. forward sanity check
4. debug-mode training
5. full training

That order isolates shape bugs before optimizer behavior enters the picture.

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

## Training Workflow Worked Notes

The parser training code has two separate responsibilities:

- `train_for_epoch`: run minibatches and update parameters
- `train`: manage epochs, evaluation, and reporting

The basic minibatch loop should be:

```python
model.train()

for train_x, train_y in minibatches:
    optimizer.zero_grad()
    logits = model(train_x)
    loss = loss_func(logits, train_y)
    loss.backward()
    optimizer.step()
```

The ordering matters. `optimizer.zero_grad()` must happen before `backward()` so gradients from previous minibatches do not accumulate. `optimizer.step()` must happen after `backward()` because the optimizer needs current gradients.

### Targets For CrossEntropyLoss

`CrossEntropyLoss` expects:

```text
logits shape: B x 3
targets shape: B
```

The target tensor should contain class indices, not one-hot vectors. If `train_y` is one-hot, the loss call is wrong for this assignment setup. If logits are already softmaxed, the loss call is also wrong because `CrossEntropyLoss` applies the log-softmax internally.

The correct mental model:

```text
model output: raw class scores
loss function: log-softmax + negative log likelihood
target: integer class id
```

### Train Mode And Eval Mode

The mode switch controls dropout:

```python
model.train()
```

enables dropout during training.

```python
model.eval()
```

disables dropout during evaluation.

When evaluating dev or test UAS, gradients are not needed:

```python
model.eval()
with torch.no_grad():
    dev_UAS = parser.parse(dev_data, model)
```

The exact utility call depends on the starter code, but the principle is stable: evaluation should be deterministic and should not build a gradient graph.

### Debug Mode Before Full Training

The handout gives a debug mode:

```bash
python run.py -d
```

The expected rough target is:

```text
loss < 0.2
dev UAS > 65
```

This does not prove the full parser is correct, but it catches major implementation mistakes quickly. If debug mode cannot learn the small subset, full training is wasted time.

After debug mode is stable, full training is:

```bash
python run.py
```

The rough full-training target is:

```text
train loss < 0.08
dev UAS > 87
```

The model from the original neural dependency parsing paper is around 92.5 UAS, so the assignment target is lower than the paper result but high enough to catch broken implementations.

### What UAS Measures

UAS means Unlabeled Attachment Score. It is:

```text
number of predicted dependencies with the correct head
/
number of total dependencies
```

"Unlabeled" means the dependency relation type is ignored. If the parser attaches `conference` to the correct head but gives the wrong relation label, UAS still counts the head as correct. This assignment's model predicts transitions, not dependency labels, so UAS is the appropriate metric.

### Debugging Order

When the parser fails, I should avoid jumping straight to optimizer tuning. The failure order should be:

1. Transition mechanics: `parser_transitions.py part_c`
2. Minibatch parsing: `parser_transitions.py part_d`
3. Embedding lookup: `parser_model.py -e`
4. Forward pass: `parser_model.py -f`
5. Debug training: `run.py -d`
6. Full training: `run.py`

Common symptom mapping:

| Symptom | Likely source |
| --- | --- |
| parse never terminates | completed-parse filtering or illegal transition handling |
| dependency heads reversed | `LEFT-ARC` / `RIGHT-ARC` implementation swapped |
| shape error in first linear layer | embedding lookup flattening or `input_size` |
| loss does not decrease in debug mode | logits/loss mismatch, no optimizer step, dropout mode issue |
| dev UAS changes between repeated evals | dropout still active during evaluation |
| grading timeout | slow embedding lookup or inefficient minibatch parsing |

### Checkpoint 6 Postmortem

The parser is conceptually small:

```text
parser state -> token ids -> embeddings -> hidden layer -> transition logits
```

Most of the difficulty is not model depth. It is respecting contracts between files: feature extraction returns token ids, the model returns logits, the loss expects class indices, and minibatch parsing expects completed parses to be filtered without losing original order.

## Final A2 Evaluation Notes

The final parser result should be read through UAS and error analysis, not just through whether the training script runs.

The handout's rough targets are:

```text
debug mode: loss < 0.2 and dev UAS > 65
full mode:  train loss < 0.08 and dev UAS > 87
```

These are sanity targets. They tell me the implementation is probably wired correctly, but they do not prove the parser understands all syntax. A transition-based parser can do well on common local patterns and still make systematic mistakes on long attachments, coordination, prepositional phrase attachment, or unusual sentence structure.

### Reading Parser Errors

When a parse is wrong, I should ask which piece of the system made the mistake:

| Error pattern | What I should inspect |
| --- | --- |
| wrong head for a nearby dependent | local feature representation or transition classifier |
| prepositional phrase attached too high or too low | parser's limited lookahead and ambiguity in stack/buffer state |
| coordination handled badly | feature templates may not capture enough structure for parallel phrases |
| ROOT attached to the wrong word | final transition decisions or learned preference for sentence heads |
| many impossible parses | transition legality, minibatch filtering, or stack mutation bug |

The model predicts transitions greedily. That means an early bad transition changes the future parser state. Later predictions can be locally reasonable given a bad state but still produce a globally wrong tree. This is different from a graph-based parser that scores complete dependency structures more directly.

### What The Reported Scores Mean

If the model reaches the target dev UAS, I should report both dev and test UAS in the written submission. The dev score is useful while debugging and tuning. The test score is the final held-out estimate.

I should not overstate small differences. A dev UAS of 88 versus 89 is less important than knowing whether the implementation is correct, whether dropout is disabled during evaluation, and whether the same run is reasonably reproducible. The handout already says there is some randomness in training.

### What I Would Save In A Debug Log

If I were running the starter code locally, these are the notes worth saving after each run:

```text
command:
debug/full:
train loss:
dev UAS:
test UAS:
seed if controlled:
code changes since last run:
failure symptoms:
next hypothesis:
```

This prevents random optimizer behavior from turning into vague memory. It also makes it easier to tell whether a change improved the parser or just changed the run.

## Final A2 Postmortem

A2 connects three levels of the course:

1. Word2Vec derivatives show the mechanics of softmax, cross-entropy, and vector updates.
2. Adam and dropout show general training tools for neural networks.
3. The dependency parser applies those tools to a structured NLP problem.

The main mathematical pattern is the softmax-cross-entropy gradient:

```text
prediction - target
```

It appears first as:

```text
yhat - y
```

for Word2Vec outside-word prediction. It appears again as:

```text
yhat_i - y_i
```

for parser transition classification. The vocabulary-sized classifier and the three-class parser classifier are the same idea at different output sizes.

The main implementation lesson is that shape contracts matter more than the apparent complexity of the model. The parser is not deep, but it has several strict interfaces:

- `PartialParse` must maintain stack, buffer, and dependency state correctly.
- `minibatch_parse` must preserve original sentence order while filtering completed parses.
- feature extraction produces fixed-length token-id lists.
- embedding lookup turns those ids into one flat vector per example.
- the model returns logits, not probabilities.
- `CrossEntropyLoss` receives class indices, not one-hot labels.
- evaluation disables dropout.

If any one of those contracts is violated, the code can still run while training the wrong thing.

## What I Am Counting As Complete

For this repository, A2 is complete as a study-notes milestone. I have not mirrored Stanford's starter archive or committed assignment solution code here. The completed work is the path I wanted for this independent study:

- derive the Word2Vec gradients carefully
- understand Adam and dropout before using them
- work through parser transitions by hand
- map the parser starter files to their responsibilities
- document tensor shapes from embedding lookup through logits
- write down the training and debugging workflow
- understand how UAS and parser errors should be interpreted

The next assignment should build from this: A3's self-attention and Transformers will be easier if I keep the same habits from A2, especially shape tracking, explicit objectives, and not treating framework calls as magic.

## Current Status

Complete as a study-notes milestone. I have worked through the Word2Vec derivatives, Adam, dropout, transition-system mechanics, parser model, embedding lookup, training loop, debugging workflow, UAS interpretation, and final postmortem.
