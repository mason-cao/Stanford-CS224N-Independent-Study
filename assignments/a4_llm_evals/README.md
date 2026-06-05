# A4: LLM Benchmarking and Evaluation

## Official Slot

- Course week: Week 5 release, Week 7 due
- Official handout: <https://web.stanford.edu/class/cs224n/assignments_w26/a4.pdf>
- Official code release: <https://web.stanford.edu/class/cs224n/assignments_w26/a4.zip>
- Related lecture: <https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture11-evaluation.pdf>

## Repo Status

Started.

I have started the A4 study pass by mapping the assignment structure: standard GSM8K benchmarking, AlpacaEval LLM-as-judge evaluation, and authorized red-team evaluation of course-provided models.

The assignment-specific notes are in [notes.md](notes.md).

## Current Study Focus

- Treat evaluation as a pipeline, not just a number.
- Separate model behavior from prompt, extractor, judge, and aggregation behavior.
- Keep raw model responses and judge outputs so later analysis is reproducible.
- Watch for judge bias, especially preference for longer outputs.
- Keep the red-team section inside the course-provided sandbox and follow the handout's honor-code boundary.

## Checkpoint Plan

1. Standard benchmarking: GSM8K string-match accuracy for models A and B, plus a prompt-improvement pass for model A. Started.
2. LLM-as-judge benchmarking: AlpacaEval pairwise comparison of models E and F using model Z, manual audit, and length-bias analysis. Not started.
3. Red-team evaluation: authorized prompt attempts for models G, H, and I, verified with `test_password`, with failed strategies preserved for model I. Not started.

## Intended Local Structure

- `code/` for starter files once pulled locally
- `notes.md` for assignment-specific evaluation notes
- `benchmarking-log.md` for model scores, prompt variants, and cached-output decisions
- `judge-audit.md` for manual AlpacaEval review notes
- `redteam-log.md` for authorized prompt attempts and checker results
