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

This is my tracker for the current public CS224N assignment sequence. I am following the official course website for the assignment order, slides, and notes, and using the public Stanford lecture playlist linked from the course page when videos are useful.

| Assignment | Status |
| --- | --- |
| [ ] A1: Introduction to Word Vectors | In progress |
| [ ] A2: Word2Vec and Dependency Parsing | Not started |
| [ ] A3: Self-Attention and Transformers | Not started |
| [ ] A4: Large Language Model Benchmarking and Evaluation | Not started |

I will keep updating this as I finish each assignment, revisit the lecture material, and write up the parts that are mathematically interesting or practically non-obvious.

Since CS224N assignments change across offerings, this tracker follows the current public course website rather than older archived versions.

## Technical Focus

Right now I am starting with word vectors, because that feels like the right place to build real intuition before jumping into bigger models. I want to understand what it actually means for words to live in a vector space: what the training objective is doing, why similar contexts lead to similar representations, and where the geometry is useful versus where it is just a nice visualization.

Later in the course, I am especially interested in the path from RNNs to Transformers. But I do not want to rush there. For now, the goal is to get the basics of representation learning solid enough that the later models feel like extensions of ideas I already understand, not a pile of architecture diagrams.

## Credits

- Official Stanford CS224N course site: [CS224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/)
- Official Stanford course schedule and materials: [CS224N Course Schedule](https://web.stanford.edu/class/cs224n/#schedule)
- Official public CS224N lecture videos linked by the course site: [Stanford CS224N YouTube playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rOaMFbaqxPDoLWjDaRAdP9D)
