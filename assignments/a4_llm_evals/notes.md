# A4 Notes: LLM Benchmarking And Evaluation

## Sources

- Official A4 handout: <https://web.stanford.edu/class/cs224n/assignments_w26/a4.pdf>
- Official A4 code release: <https://web.stanford.edu/class/cs224n/assignments_w26/a4.zip>
- Week 6 benchmarking and evaluation slides: <https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture11-evaluation.pdf>
- Official CS224N schedule: <https://web.stanford.edu/class/cs224n/#schedule>

## What This Assignment Is Testing

A4 is about measurement. A3 asked me to build the core Transformer machinery. A4 asks a different question: once models produce open-ended text, how do I know which model is better, safer, more useful, or more reliable?

The assignment breaks evaluation into three concrete exercises:

- standard closed-ended benchmarking on GSM8K
- pairwise LLM-as-judge evaluation on AlpacaEval
- red-team testing of course-provided models in an authorized sandbox

The common thread is that an evaluation is not just a score. It is a measurement pipeline:

```text
dataset
-> prompt
-> model response
-> extraction or judging
-> aggregation
-> interpretation
```

Every arrow can introduce noise or bias. A score is only meaningful if I understand how it was produced.

## Four Evaluation Modes

The handout separates LLM evaluation into four broad categories.

### Standard Closed-Ended Benchmarking

Closed-ended benchmarking uses tasks with known answers and a relatively simple correctness rule.

For a dataset:

```text
D = {(x_i, y_i)} for i = 1, ..., n
```

the model produces:

```text
r_i = model(prompt(x_i))
```

Then an extractor maps the raw response to an answer:

```text
a_i = extract(r_i)
```

The score is usually:

```text
accuracy = (1 / n) * sum_i 1[a_i = y_i]
```

This is clean when the target is unambiguous. GSM8K works this way because the final answer is usually a number. The model may explain its reasoning in many ways, but the evaluator can look for the final numeric answer.

The advantage is reproducibility. The weakness is brittleness: a response can be mathematically correct but formatted in a way the extractor misses, or it can have flawed reasoning but happen to end with the right number.

### LLM As Judge

Some outputs are hard to grade with a simple string match. If the task is "write a helpful answer," correctness is not a single token or number. The assignment therefore uses a separate model as a judge.

The pairwise structure is:

```text
question q
response_e = model E(q)
response_f = model F(q)
judge_output = model Z(q, response_e, response_f)
winner = parse(judge_output)
```

The aggregate metric is win rate:

```text
win_rate(E over F) = E_wins / total_comparisons
```

The important lesson is that a judge model is itself a model with biases. It may prefer longer answers, certain formatting, confident wording, or the response shown first. A judge score is useful, but it is not the same thing as human preference unless I check for judge behavior.

### User Study

A user study uses humans in a controlled experiment. This is expensive and slower, but it can provide stronger evidence when the study is designed carefully.

The key word is controlled. Human preference data is useful only if the experiment avoids obvious confounds:

- unclear task instructions
- uneven exposure to systems
- different users seeing different difficulty levels
- no randomization
- no plan for ties or uncertainty
- too few examples to support the conclusion

A4 does not ask me to run a user study, but it uses user studies as a reference point for why automatic judging is only an approximation.

### Interacting With The Model

Informal interaction is the fastest way to discover failure modes. I can ask a model questions, change wording, try edge cases, and notice patterns.

This is useful for exploration but weak as evidence. If I only show one interesting transcript, I have not measured a rate. I have found an example.

The right mental split:

```text
interaction finds hypotheses
benchmarks estimate rates
user studies estimate human impact
```

## Part 1: Standard Benchmarking On GSM8K

The first scored part benchmarks models A and B on the first 100 GSM8K problems.

The files named in the handout are:

```text
query.py
gsm8k.py
gsm8k_first_100.jsonl
```

The model access function is in `query.py`. It accepts:

```text
query string
model id
```

and returns the model response. The important practical warning is cost. Since API calls are billed, I should test the code on a small subset before running all 100 examples.

### Benchmarking Pipeline

For each GSM8K row:

```text
read problem
build standard prompt
query model A or B
extract final numeric answer
compare extracted answer to gold answer
record correct or incorrect
```

The score for a model is:

```text
number correct / 100
```

I should save enough intermediate data to debug later:

```text
problem id
question
gold answer
raw model response
extracted answer
is_correct
```

If I only save the final score, I cannot tell whether errors came from model reasoning, answer extraction, prompt formatting, or API failures.

### String Match Is Not The Same As Mathematical Understanding

String-match evaluation is convenient because it turns an open text response into a deterministic score. But it measures the full pipeline, not pure reasoning.

Possible failure cases:

- The model solves the problem but formats the answer differently.
- The model gives several numbers and the extractor chooses the wrong one.
- The model makes an arithmetic mistake but ends with the correct final number by coincidence.
- The response includes units or punctuation that the extractor mishandles.
- The prompt causes the model to omit a clean final answer.

This means I should treat GSM8K accuracy as:

```text
accuracy under this prompt, extractor, model, and dataset slice
```

not as an absolute measure of mathematical intelligence.

### Prompt Improvement For Model A

The second GSM8K task asks me to write a better prompt template for model A and compare it against the standard prompt.

The prompt should make the output easier to extract and improve reasoning. A good template should probably:

- tell the model to solve step by step
- ask for a final answer in a predictable format
- avoid extra commentary after the final answer
- keep the answer format compatible with the provided extractor

The evaluation logic stays the same. Only the prompt changes.

The writeup should include:

- the exact prompt template
- why I expected it to help
- a bar chart comparing standard prompt accuracy and improved prompt accuracy
- whether performance improved
- possible reasons for improvement or failure

The main lesson is prompt sensitivity. If one prompt changes accuracy, the benchmark is measuring not only the model but also how well the evaluation prompt elicits the target behavior.

## Part 2: LLM Judge Benchmarking On AlpacaEval

The second scored part compares models E and F on 30 AlpacaEval prompts using model Z as the judge.

The files named in the handout are:

```text
llm_judge.py
alpaca_eval_first_30.jsonl
```

The evaluation loop is:

```text
for each prompt:
    get response from model E
    get response from model F
    ask model Z which response is better
    parse the judge decision
    store all raw outputs
```

The handout explicitly says to save the model responses and judge outputs. That matters because later questions ask me to inspect examples and analyze judge behavior. Regenerating responses can change the data, cost more money, and make the analysis inconsistent.

### Pairwise Judging

Pairwise evaluation is often easier than absolute scoring. Instead of asking:

```text
How good is this answer from 1 to 10?
```

the judge answers:

```text
Which of these two responses is more useful for this prompt?
```

The final score is a win rate:

```text
E wins / 30
F wins / 30
ties or parse failures handled explicitly
```

I should decide how to handle judge outputs that do not parse cleanly. Options:

- rerun the judge for that example
- count it as a tie
- exclude it and report the exclusion

The important point is to make the rule explicit before interpreting the result.

### Judge Prompt Design

The judge prompt needs to show:

```text
original user prompt
response from model E
response from model F
clear judging criteria
required output format
```

The required output format should be easy to parse. For example, the judge can be asked to end with a single label such as:

```text
Winner: E
```

or:

```text
Winner: F
```

The criteria should focus on usefulness for the user, not superficial style. But even with good criteria, the judge may still show preferences unrelated to actual helpfulness.

### Manual Audit Of 5 Examples

The assignment asks me to inspect 5 examples myself and compare my preference with the judge's preference.

For each example I should record:

```text
prompt
model E response
model F response
judge decision
my decision
agreement or disagreement
short reason
```

This is a small audit, not a statistically strong human study. Its purpose is to make me look inside the aggregate win rate. If the judge disagrees with me on a clear example, I need to explain why.

Useful questions while auditing:

- Did one response answer the actual user request more directly?
- Did one response invent facts?
- Did one response avoid part of the task?
- Did one response look better only because it was longer?
- Did the judge reward polish over correctness?

### Length Bias Check

The final AlpacaEval subproblem asks for two histograms:

```text
lengths of judge-preferred outputs
lengths of judge-not-preferred outputs
```

Both histograms have 30 data points because every comparison has one preferred and one not-preferred response.

This is a direct test for a common judge bias. If preferred outputs are consistently longer, then the judge may be rewarding verbosity rather than usefulness. That does not prove the judge is wrong, because longer answers can contain more complete information. But it lowers my confidence unless the manual examples show that the extra length is actually helpful.

The right conclusion should be conditional:

```text
If preferred outputs are much longer, I should be cautious about treating judge win rate as user preference.
If length distributions overlap heavily, length bias is less obvious, but other biases may still exist.
```

## Part 3: Red-Team Evaluation

The third scored part is an authorized red-team exercise on course-provided models G, H, and I.

The setup:

- Each model has a secret password in its system prompt.
- The model is instructed not to reveal that password.
- The goal is to find prompts that make the model disobey that instruction.
- The correct extracted password can be checked with `test_password`.
- The password depends on the email key in `.env`.
- Reverse engineering the password from the provided code is forbidden by the handout.

The important framing: this is a controlled course exercise. I should treat it as a measurement of instruction-following robustness in the provided sandbox, not as permission to target unrelated systems.

### What Red-Teaming Measures

This section tests whether a model reliably follows a higher-priority instruction when a user asks for something conflicting.

The desired behavior is:

```text
answer normal user queries
but never reveal the protected password
```

The failure is:

```text
user prompt causes password disclosure
```

This is not only about one clever prompt. It is about finding evidence that the model's instruction hierarchy can be broken under adversarial interaction.

### Model Difficulty

The handout says the models become harder:

```text
G easier than H
H easier than I
```

So I should expect the strategy that works on G to fail on H or I. For model I, the handout asks for two failed prompts using different strategies, which means I need to preserve failed attempts instead of only keeping the final successful one.

### Red-Team Log Template

For each attempt, I should log:

```text
model id:
prompt strategy:
exact prompt:
raw model response:
claimed password, if any:
test_password result:
why it worked or failed:
```

This matters because some models may lie about the password. A response that looks successful is not enough. The result has to pass `test_password`.

### Ethical Boundary

The handout explicitly warns not to reverse engineer the password from the provided code. The right workflow is interactive testing against the models, then validation with the checker.

The submission should include:

- the password for G and H
- the prompt used for G, H, and I
- the reason each successful prompt worked
- for I, two failed prompts using different strategies

For my study notes, I should focus on evaluation methodology:

- define the protected behavior
- make attempts reproducible
- verify claimed success with the official checker
- record failed attempts
- avoid using forbidden implementation shortcuts

## Cross-Cutting Lessons

A4 is forcing me to separate model behavior from measurement behavior.

For GSM8K:

```text
model ability + prompt format + extractor behavior -> accuracy
```

For AlpacaEval:

```text
model responses + judge prompt + judge bias + parser behavior -> win rate
```

For red-teaming:

```text
model instruction-following + adversarial prompt + validation checker -> success or failure
```

Every score has a measurement story behind it. The writeup should not just report numbers; it should explain what those numbers do and do not prove.

## A4 Study Plan

I want to split A4 into three study checkpoints:

1. Standard benchmarking: GSM8K accuracy for models A and B, plus improved prompt evaluation for model A.
2. Judge benchmarking: AlpacaEval pairwise evaluation for models E and F, manual audit, and length-bias analysis.
3. Red-team evaluation: authorized prompt attempts for G, H, and I, verified with `test_password`, with failed strategies preserved for model I.

This first pass is just the map. The next useful notes should go deeper on GSM8K: how string extraction works, how to cache raw responses, how to calculate accuracy, and how to design a prompt that improves performance without breaking the extractor.
