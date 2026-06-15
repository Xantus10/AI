# LLM

## Models

### GPTv1

The simplest GPT. Features only a single Embedding layer.

### GPTv2

Architecture inspired by GPT2 and nanoGPT. This model works on principle of self-attention.

Architecture uses

- Token and positional embeddings
- Passed through LLM blocks which consist of
    - Paralelized heads of self-attention
    - FeedForward system
    - With a Skip connection through the block
- And a linear layer mapping back to tokens

## Tokenizers

### Character

The simplest tokenizer. Just goes through the text character by character.

### FreqBigram

Character-wise tokenizer + adds `N` most frequent bigrams to the vocabulary.
