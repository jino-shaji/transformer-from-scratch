# How a Transformer Actually Works: A Tiny Worked Example

Transformers are everywhere now, but most explanations stop at the buzzwords. The real idea is simple: a model can look at every word in a sentence at once and decide which ones matter most.

That is the core of self-attention, and it is the reason transformers became the foundation of modern language models.

## Why transformers changed everything

Before transformers, many language models relied on sequential processing. Recurrent neural networks and LSTMs read one token at a time, which made it harder to capture long-range context.

The transformer introduced a different approach. Instead of reading words one by one, it lets each word compare itself with every other word in the sentence. That means the meaning of a word can be updated using the surrounding context immediately.

This is what makes transformers so powerful for tasks like translation, summarization, question answering, and chat generation.

## A tiny example with actual math

Consider the phrase: "river bank".

If you read that sentence, you know the word "bank" probably refers to the edge of a river, not a financial institution. The transformer learns this by paying attention to the word "river".

Let us make the example very small so the math is easy to follow.

### 1. Words become vectors

We represent the words as simple vectors:

```text
river = [1, 0]
bank  = [0, 1]
```

### 2. Each word creates Query, Key, and Value vectors

Every word produces three vectors:

- Query: what the word is looking for
- Key: what the word offers
- Value: the information the word contributes

In our toy example, these are produced by multiplying the word vectors by learned weight matrices.

```text
W_Q = [[0, 1],
       [1, 0]]

W_K = [[1, 0],
       [0, 1]]

W_V = [[2, 0],
       [0, 1]]
```

For the word "bank", we get:

```text
q_bank = [0, 1] × W_Q = [1, 0]

k_river = [1, 0] × W_K = [1, 0]
k_bank = [0, 1] × W_K = [0, 1]
```

### 3. Score the words against the query

We compare the query of "bank" with the keys of "river" and "bank" itself using a dot product:

```text
bank → river: [1, 0] · [1, 0] = 1
bank → bank:  [1, 0] · [0, 1] = 0
```

This tells us that "bank" strongly matches "river" and not itself.

### 4. Convert scores into attention weights

These scores are scaled and passed through softmax. The result gives the model a probability distribution over the surrounding words.

```text
scaled = [0.71, 0]
softmax = [0.67, 0.33]
```

So the model gives 67% of the attention to "river" and 33% to itself.

### 5. Blend the values

Now the model combines the values of the words according to those weights.

```text
v_river = [1, 0] × W_V = [2, 0]
v_bank = [0, 1] × W_V = [0, 1]

output = 0.67 × [2, 0] + 0.33 × [0, 1]
       = [1.34, 0.33]
```

That updated vector is the new representation of "bank" after considering context. This is the heart of attention.

## What makes the real transformer different

The toy example shows the essential idea. Real transformers scale this mechanism in several important ways:

1. Multi-head attention
   - The model runs many attention operations in parallel.
   - Different heads can focus on different linguistic patterns.

2. Feed-forward layers
   - After attention, each token is passed through a small neural network.

3. Positional encoding
   - Attention itself does not know word order, so transformers add position information.

4. Stacking many layers
   - Multiple transformer blocks are stacked to build richer representations.

## Why this matters

The big breakthrough was not just the attention mechanism itself. It was the fact that attention can be computed in parallel. That made training large models practical on modern hardware.

That parallelism is one of the main reasons transformers became so successful. They are now behind large language models, coding assistants, image generators, and many other AI systems.

## Final takeaway

A transformer does not read text like an RNN. It lets each token look at all the others and decide what matters. That simple shift changed AI.

If you understand attention, you understand the core idea behind modern language models.

---

## Suggested category

Category: Artificial Intelligence / Machine Learning

## Suggested tags

- transformers
- attention
- neural networks
- ai
- machine learning
