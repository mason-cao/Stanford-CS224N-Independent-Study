# Week 1 Notes: Word Vectors

## Sources

- CS224N schedule and Week 1 materials: <https://web.stanford.edu/class/cs224n/#schedule>
- Official public CS224N playlist: <https://www.youtube.com/playlist?list=PLoROMvodv4rOaMFbaqxPDoLWjDaRAdP9D>

## What Actually Clicked For Me

The main conceptual jump is that a word vector is not "meaning" in the abstract. It is a trainable compromise that becomes useful because the model is forced to predict context. That makes the geometry downstream of the objective function, not some magical semantic lookup table.

I also think it is easy to say "distributional hypothesis" without really cashing it out. In practice, the useful claim is narrower: words that appear in similar local environments end up with related representations because the model can reuse parameters across contexts. The embedding space is a side effect of that compression pressure.

## The Core Idea I Want To Keep Straight

The main idea is to stop treating words as isolated symbols. A one-hot vector can identify a word, but it does not say anything about how that word relates to other words. A dense vector is more useful because distance and direction can start to carry information about usage.

The part I want to keep clear is that the geometry is learned from context. Words end up near each other because the training signal rewards similar behavior in similar environments. That does not mean the vector "knows" meaning in a human sense, but it gives the model a much better starting point than raw word IDs.

I also want to be careful not to overread pretty embedding examples. Analogies and nearest neighbors are useful checks, but they are not the same thing as understanding the whole representation.

## Things I Do Not Fully Trust Yet

- I understand the high-level distributional idea, but I still need more practice connecting it to the actual training setup.
- I know analogies are a neat qualitative demo, but I do not yet trust them as serious evidence that an embedding space is "good."
- I want to run more nearest-neighbor inspections before I claim I have geometric intuition rather than slogan-level intuition.

## Why This Matters For The Rest Of The Course

This week sets up a pattern that keeps showing up later:

- define a prediction problem
- learn dense vectors because discrete symbols do not compose well
- let the training objective shape the representation geometry

That logic does not stop with static word vectors. It scales all the way into contextual encoders and autoregressive language models.

## Repo Connection

This is the first set of notes I am using for A1. I have not started the later assignments yet.
