# Week 2 Notes: Neural Networks And Backpropagation

## Sources

- CS224N schedule and Week 2 materials: <https://web.stanford.edu/class/cs224n/#schedule>
- CS224N Lecture 3 slides: <https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture03-neuralnets.pdf>
- CS224N neural networks and backpropagation notes: <https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes03-neuralnets.pdf>
- A2 handout: <https://web.stanford.edu/class/cs224n/assignments_w26/a2.pdf>

## Why This Unit Matters

The word-vector unit gave me the representation-learning setup: choose an objective, update dense vectors, and inspect the geometry that results. Week 2 shifts the emphasis from embeddings as objects to neural networks as trainable functions. The core question becomes: once a model has multiple layers of parameters, how do I compute all the gradients correctly and efficiently?

Backpropagation is the practical answer. It is not a separate learning algorithm from gradient descent. It is the efficient bookkeeping method that uses the chain rule to compute gradients for every intermediate variable and parameter in a computational graph.

The reason this matters for the rest of CS224N is direct:

- Word2Vec gradients in A2 require careful vector calculus.
- The dependency parser in A2 is a small neural network trained end to end.
- RNNs, attention, and Transformers are all larger computational graphs, but the training logic is the same chain-rule machinery.

If I am shaky on shapes or local derivatives here, later architectures will look more mysterious than they need to.

## From Linear Classifiers To Neural Networks

A linear classifier computes scores like:

```text
s = W x + b
```

where:

- `x` is the input feature vector
- `W` maps input features to class scores
- `b` shifts the scores
- `s` contains one score per class

The limitation is that the decision boundary is linear in the input representation. If the representation already makes the task linearly separable, this is enough. If not, a linear classifier has no way to invent more useful intermediate features.

A one-hidden-layer neural network adds a learned intermediate representation:

```text
h = f(W1 x + b1)
s = W2 h + b2
```

The nonlinearity `f` is essential. Without it, the composition collapses:

```text
W2 (W1 x + b1) + b2
```

is still just an affine function of `x`. Stacking linear maps without nonlinearities does not buy expressive power. The network becomes useful because the hidden layer can learn nonlinear feature detectors, and the output layer can combine those learned features for the task.

For NLP, the input `x` is often made by concatenating or combining word vectors. In the simple window classification example from lecture, a center word and its surrounding context vectors are concatenated into one long vector:

```text
x_window = [x_{t-2}; x_{t-1}; x_t; x_{t+1}; x_{t+2}]
```

If each word vector has dimension `d` and the window has 5 words, then:

```text
x_window in R^{5d}
```

That shape discipline matters. A large fraction of neural network bugs are really shape bugs.

## Softmax Classifier Refresher

For multiclass classification, the model produces raw scores:

```text
s = W h + b
```

Softmax converts scores into a probability distribution:

```text
yhat_i = exp(s_i) / sum_j exp(s_j)
```

Cross-entropy for a one-hot label `y` is:

```text
J = - sum_i y_i log(yhat_i)
```

If the true class is `o`, then only `y_o = 1`, so:

```text
J = -log(yhat_o)
```

This is the same simplification that appears in Word2Vec. The true distribution is one-hot, so the cross-entropy just picks out the negative log probability assigned to the correct class.

The most important derivative pattern:

```text
dJ / ds = yhat - y
```

That is the error signal at the softmax-score layer. It says:

- if a wrong class got too much probability, push its score down
- if the true class got too little probability, push its score up
- the magnitude depends on how wrong the model's probability was

This pattern shows up again in A2's naive softmax Word2Vec derivation.

## Matrix Calculus Rules I Need To Keep Straight

The course expects gradients to have the same shape as the variable being differentiated. That convention is useful because it makes update rules natural:

```text
theta := theta - alpha * dJ/dtheta
```

The gradient must be shaped like `theta`.

Useful shape patterns:

```text
z = W x + b
```

If:

```text
W in R^{m x n}
x in R^n
b in R^m
z in R^m
```

and an upstream gradient is:

```text
delta = dJ/dz in R^m
```

then:

```text
dJ/dx = W^T delta          shape R^n
dJ/dW = delta x^T          shape R^{m x n}
dJ/db = delta              shape R^m
```

This is worth memorizing, but not as a magic formula. The shape logic explains it:

- `W^T delta` maps output-space error back into input space.
- `delta x^T` gives every weight `W_ij` the product of output error `delta_i` and input activation `x_j`.
- `b` contributes directly to `z`, so its gradient is just the upstream gradient.

For a batch, the same idea becomes matrix multiplication over examples. If examples are columns:

```text
Z = W X + b
```

then the gradient for `W` accumulates outer products across the batch:

```text
dJ/dW = Delta X^T
```

The batch version is not a different derivative. It is the vectorized accumulation of many single-example derivatives.

## Backpropagation As Local Chain Rule

A computational graph breaks a complicated function into simple operations. During the forward pass, each node computes a value. During the backward pass, each node receives an upstream gradient and sends gradients to its inputs.

The local rule is:

```text
downstream gradient = upstream gradient * local derivative
```

For vector-valued nodes, this becomes matrix multiplication with the relevant Jacobian. In practice, I should not build full Jacobians unless I need them conceptually. Most operations have compact vector-Jacobian products.

Example:

```text
z = W x + b
h = tanh(z)
s = U h + c
J = cross_entropy(softmax(s), y)
```

Backward pass:

```text
delta_s = yhat - y
dJ/dU = delta_s h^T
dJ/dc = delta_s
dJ/dh = U^T delta_s
delta_z = dJ/dh * (1 - tanh(z)^2)
dJ/dW = delta_z x^T
dJ/db = delta_z
dJ/dx = W^T delta_z
```

The key idea is that every layer only needs:

- the upstream gradient from the layer above
- the local values cached during the forward pass
- the local derivative of its own operation

That is why forward-pass caching matters in implementations.

## Activation Functions

Nonlinear activations let neural networks learn non-linear functions. The main ones I need to distinguish:

### Sigmoid

```text
sigma(z) = 1 / (1 + exp(-z))
```

Derivative:

```text
sigma'(z) = sigma(z) * (1 - sigma(z))
```

Sigmoid squashes values into `(0, 1)`, which can be useful for binary probabilities, but it saturates. When `z` is very positive or very negative, the derivative gets small and gradients shrink.

### Tanh

```text
tanh(z)
```

Derivative:

```text
1 - tanh(z)^2
```

Tanh is zero-centered, which often makes it nicer than sigmoid for hidden states, but it also saturates at large magnitudes.

### ReLU

```text
ReLU(z) = max(0, z)
```

Derivative:

```text
1 if z > 0
0 if z < 0
```

ReLU avoids saturation on the positive side and is cheap to compute. The failure mode is dead units: if a unit is stuck in the negative region, its gradient can be zero for many examples.

## Gradient Checking

Gradient checking compares the analytical gradient from backpropagation to a numerical finite-difference estimate:

```text
dJ/dtheta_i ~= (J(theta_i + epsilon) - J(theta_i - epsilon)) / (2 epsilon)
```

This is too slow for training but useful for debugging implementations. If a hand-derived gradient is wrong, gradient checking catches it before the optimizer hides the mistake inside noisy training behavior.

What I need to remember:

- use a small `epsilon`, but not so small that floating-point error dominates
- compare relative error, not just absolute error
- run checks on tiny inputs
- turn off stochastic behavior like dropout while checking gradients

Gradient checking is especially useful for A2 because the Word2Vec derivatives are easy to get almost right while still having a transpose, sign, or shape error.

## Initialization And Learning Rate

If parameters are initialized too large, activations can saturate and gradients can behave badly. If parameters are initialized too small or symmetrically, units may learn slowly or redundantly. The goal is to start with random weights at a scale that keeps signals and gradients in a reasonable range.

The high-level Xavier intuition:

```text
variance should not explode or vanish as values move through layers
```

Learning rate controls update size:

```text
theta := theta - alpha * dJ/dtheta
```

If `alpha` is too large, training can diverge or bounce around. If it is too small, training may be painfully slow or look stuck. Adaptive optimizers such as AdaGrad and Adam change the effective learning rate per parameter based on gradient history, but they do not remove the need to understand the underlying gradient direction.

## How This Connects To A2

A2 has three conceptual pieces:

- derive Word2Vec gradients for the naive softmax loss
- understand dropout and Adam as general neural network training techniques
- implement a neural dependency parser in PyTorch

The first part is pure backpropagation through the Word2Vec softmax. The center vector `v_c` is like an input representation, the outside vector matrix `U` is like a classifier weight matrix, and the predicted distribution `yhat` is the softmax output.

The central derivative I expect to reuse:

```text
dJ/dv_c = U (yhat - y)
```

or the transpose variant depending on whether vectors are stored as columns or rows. The A2 handout stores outside vectors as columns of `U`, so I need to track the convention carefully rather than blindly copying the form from my Week 1 notes.

For the parser, the same pattern scales up:

```text
embeddings -> hidden layer -> activation -> scores -> loss
```

PyTorch will compute gradients automatically, but A2 wants me to understand what the library is doing. That is the point of doing the math before leaning on autograd.

## Things I Do Not Fully Trust Yet

- I need to redo the softmax derivative by hand until `yhat - y` feels inevitable rather than memorized.
- I need to be more careful about row-vector versus column-vector conventions; the same derivative can look transposed depending on storage.
- I understand the idea of gradient checking, but I have not yet built the habit of using it early enough.
- I need to connect dropout and Adam to actual training behavior, not just definitions.

## Repo Connection

This starts my Week 2 notes and prepares for A2. I have not pulled starter code into this repo yet.
