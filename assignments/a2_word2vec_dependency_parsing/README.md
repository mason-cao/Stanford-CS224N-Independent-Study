# A2: Word2Vec and Dependency Parsing

## Official Slot

- Course week: Week 2
- Official handout: <https://web.stanford.edu/class/cs224n/assignments_w26/a2.pdf>
- Official code release: <https://web.stanford.edu/class/cs224n/assignments_w26/a2.zip>

## Repo Status

Complete as a study-notes milestone.

I have finished the first pass through the Word2Vec derivative, optimization, dropout, transition-mechanics, parser model, training-loop, evaluation, and wrap-up questions.

The assignment-specific notes are in [notes.md](notes.md).

## Current Study Focus

- Keep the `yhat - y` gradient automatic for both Word2Vec and parser transition classification.
- Track vector and matrix shapes carefully, especially across embedding lookup and flattened parser features.
- Connect optimizer and regularization choices to actual training behavior.
- Treat parser transitions as state mutations that need exact stack and buffer discipline.
- Read UAS as a head-attachment metric and use parser errors to debug model behavior.

## Intended Local Structure

- `code/` for starter files once pulled locally
- `notes.md` for assignment-specific derivations
- `debug-log.md` for mistakes, failed runs, and shape issues
