# A2: Word2Vec and Dependency Parsing

## Official Slot

- Course week: Week 2
- Official handout: <https://web.stanford.edu/class/cs224n/assignments_w26/a2.pdf>
- Official code release: <https://web.stanford.edu/class/cs224n/assignments_w26/a2.zip>

## Repo Status

In progress.

I have finished the first pass through the Word2Vec derivative questions and am now connecting the math to the parser implementation.

The assignment-specific notes are in [notes.md](notes.md).

## Current Study Focus

- Review softmax and cross-entropy until the `yhat - y` gradient is automatic.
- Track vector and matrix shapes carefully, especially the column-vector convention in the A2 handout.
- Connect Word2Vec derivatives to the same backpropagation rules used in ordinary neural classifiers.
- Understand dropout and Adam conceptually before using them in the parser implementation.
- Keep parser implementation notes tied to the exact starter-code responsibilities instead of treating the parser as a black box.

## Intended Local Structure

- `code/` for starter files once pulled locally
- `notes.md` for assignment-specific derivations
- `debug-log.md` for mistakes, failed runs, and shape issues
