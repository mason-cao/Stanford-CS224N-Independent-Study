# Stanford CS224N Independent Study

## Why I'm Doing This

This repository is my working log for independently studying Stanford CS224N: *Natural Language Processing with Deep Learning*.

I am building this project because I want to move past high-level summaries of modern NLP and understand the machinery well enough to derive it, implement it, debug it, and explain it precisely. I do not want to treat Transformers as a black box or learn them only through API usage. I want to understand why the architecture works, what mathematical assumptions it makes, where its computational advantages come from, and how its components relate to the earlier sequence models it replaced.

The goal is to study the course the way an ambitious developer would: by combining theory, implementation, derivation, and iteration. That means working through the underlying linear algebra, matrix calculus, optimization, representation learning, and modeling choices that support modern language systems rather than stopping at surface-level familiarity.

This is an **independent study using publicly available course materials**. It is **not** affiliated with, enrolled in, or officially endorsed by Stanford University or the CS224N course staff.

## Learning Goals

- Rebuild core NLP intuition from first principles rather than from secondary summaries.
- Understand backpropagation, gradient flow, optimization, and representation learning well enough to reason through implementation bugs and training behavior.
- Study why recurrent architectures struggle with long-range dependencies and why attention-based methods changed the field.
- Master the internal logic of self-attention and Transformer blocks at the level of equations, tensor operations, and computational tradeoffs.
- Implement the core CS224N assignments with enough rigor to explain every objective, tensor shape, and modeling decision.
- Build the habit of learning research-level material through code, derivations, experiments, and careful writeups.

## Assignment Progress

This is my tracker for the current public CS224N assignment sequence.

| Assignment | Status |
| --- | --- |
| [ ] A1: Introduction to Word Vectors | Upcoming |
| [ ] A2: Neural Network Foundations, Tensor Derivatives, and Dependency Parsing | Upcoming |
| [ ] A3: Self-Attention and Transformers | Upcoming |
| [ ] A4: Large Language Model Benchmarking and Evaluation | Upcoming |

I plan to keep updating this as I finish each assignment, revisit the lecture material, and write up the parts that are mathematically interesting or practically non-obvious. Since CS224N assignments change across offerings, this tracker follows the current public course website rather than older archived versions.

## Technical Focus

One of the main arcs of this study is the transition from **Recurrent Neural Networks (RNNs)** to **self-attention** and the **Transformer architecture**.

I want this repository to make that transition explicit. RNNs are important because they frame sequence modeling in a way that exposes both the strengths and the limitations of recurrence: hidden-state compression, vanishing gradients, sequential computation, and difficulty with long-range dependencies. Transformers matter because self-attention changes that formulation. Instead of forcing information through a single evolving state, tokens interact directly through learned query, key, and value projections, which changes both the optimization landscape and the computational profile of the model.

That shift is a major reason I am studying CS224N in the first place. I want to understand the architecture mathematically, not just use it. That means paying close attention to word representation geometry, matrix calculus, backpropagation, attention as differentiable memory access, and the internals of Transformer blocks, pretraining objectives, and adaptation methods.

## Study Workflow

I plan to work through the course material in a consistent loop:

1. Read lecture slides, notes, and primary papers.
2. Re-derive the main equations and implementation details.
3. Implement the relevant models or assignment components.
4. Validate behavior empirically and inspect failure modes.
5. Record concise notes on intuition, design tradeoffs, and lessons learned.

## Credits

- Official Stanford CS224N course site: [CS224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/)
- Official Stanford course schedule and materials: [CS224N Course Schedule](https://web.stanford.edu/class/cs224n/index.html#schedule)
- Chris Manning lectures: [Chris Manning Lectures and Talks](https://www.youtube.com/@ChrisManningTV)
