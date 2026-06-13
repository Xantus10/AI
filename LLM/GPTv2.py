import torch
import torch.nn as nn
from torch.nn import functional as FUN

### Constants
# Size of training blocks
BLOCK_SIZE = 128
# Paralel processing (efficiency)
BATCH_SIZE = 16
# Learning steps
STEPS = 1000
# Learning rate
LEARNING_RATE = 6e-4
# Enable running on GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# Size of embedding vectors
N_EMBEDDING_DIMENSIONS = 192
# Number of heads to use
NO_OF_HEADS = 6
# Number of inner LLM Blocks
NO_OF_LLM_BLOCKS = 6
# Dropout rate
DROPOUT = 0.2


class Head(nn.Module):
  """
  Head of self attention
  """
  def __init__(self, head_size: int):
    """
    Head of self attention

    Parameters
    ----------
    - head_size: Size of the final layer in head
    """
    super().__init__()
    # What am I
    self.key = nn.Linear(N_EMBEDDING_DIMENSIONS, head_size, bias=False)
    # What am I looking for
    self.query = nn.Linear(N_EMBEDDING_DIMENSIONS, head_size, bias=False)
    # What do I provide
    self.value = nn.Linear(N_EMBEDDING_DIMENSIONS, head_size, bias=False)
    # Mask for lookup only in past
    self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
    self.dropout = nn.Dropout(DROPOUT)
  
  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x)
    q = self.query(x)
    # Default attention scores for positions (q@k will be high for aligned queries and values)
    weights = q @ k.transpose(-2, -1) * pow(C, -0.5)
    # Fill the forbiden future with -inf
    weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    # Normalize the affinities
    weights = FUN.softmax(weights, dim=-1)
    weights = self.dropout(weights)
    v = self.value(x)
    # Multiply {which positions are interested in me} with {what do I provide}
    ret = weights @ v
    return ret

### MultiHead

class MultiHead(nn.Module):
  """
  A paralelized group of self-attention heads
  """
  def __init__(self, num_of_heads: int, head_size: int):
    """
    A paralelized group of self-attention heads

    Parameters
    ----------
    - num_of_heads: How many paralel heads
    - head_size: The output dimensions of a single head
    """
    super().__init__()
    # Paralel heads of self attention
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_of_heads)])
    # Project the output from heads into N_EMBED (Mix them together)
    self.projection = nn.Linear(N_EMBEDDING_DIMENSIONS, N_EMBEDDING_DIMENSIONS)
    self.dropout = nn.Dropout(DROPOUT)

  def forward(self, x):
    # Concat outputs of all the heads
    ret = torch.cat([head(x) for head in self.heads], dim=-1)
    ret = self.projection(ret)
    ret = self.dropout(ret)
    return ret

### Single layer feed forward

class FeedForward(nn.Module):
  """
  Tokens learn what to do with their self-attention data
  """
  def __init__(self, n_embed: int):
    """
    Tokens learn what to do with their self-attention data

    Parameters
    ----------
    - n_embed: Number of embedding dimensions
    """
    super().__init__()
    self.network = nn.Sequential(
      # Expand the n_embed into 4x larger layer
      nn.Linear(n_embed, 4 * n_embed),
      # Non-linearity
      nn.ReLU(),
      # Reduce back to n_embed
      nn.Linear(4 * n_embed, n_embed),
      nn.Dropout(DROPOUT)
    )
  
  def forward(self, x):
    return self.network(x)

### A block of the LLM neural network

class LLMBlock(nn.Module):
  """
  A block of the transformer
  """
  def __init__(self, n_embed: int):
    """
    A block of the transformer

    Parameters
    ----------
    - n_embed: Number of embedding dimensions
    """
    super().__init__()
    # Self attention heads to learn token affinities
    self.self_attention_heads = MultiHead(NO_OF_HEADS, n_embed//NO_OF_HEADS)
    # Tokens learn what to do with the self-attention data
    self.feedforward = FeedForward(n_embed)
    # We normalize the data to ensure smaller values
    self.layernorm_1 = nn.LayerNorm(n_embed)
    self.layernorm_2 = nn.LayerNorm(n_embed)
  
  def forward(self, x):
    x = x + self.self_attention_heads(self.layernorm_1(x))
    x = x + self.feedforward(self.layernorm_2(x))
    return x

class GPTTransformer(nn.Module):
  """
  GPT 2 / nanoGPT like transformer
  """
  def __init__(self, vocab_size: int):
    """
    GPT 2 / nanoGPT like transformer

    Parameters
    ----------
    - vocab_size: Size of the vocabulary
    """
    super().__init__()
    # The tokens get expanded into n_embed vectors
    self.token_embedding_table = nn.Embedding(vocab_size, N_EMBEDDING_DIMENSIONS)
    # Positional information encoding
    self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBEDDING_DIMENSIONS)
    # LLM blocks
    self.llm_blocks = nn.Sequential(
      *[LLMBlock(N_EMBEDDING_DIMENSIONS) for _ in range(NO_OF_LLM_BLOCKS)],
      nn.LayerNorm(N_EMBEDDING_DIMENSIONS)
    )
    # Last layer to reduce from our n_embed vectors back into tokens
    self.lang_model_head = nn.Linear(N_EMBEDDING_DIMENSIONS, vocab_size)
  
  def forward(self, x, targets=None):
    B, T = x.shape

    token_embeddings = self.token_embedding_table(x) # format: B,T,C=N_EMB_DIM
    position_embeddings = self.position_embedding_table(torch.arange(T, device=DEVICE)) # format: T,C
    res = token_embeddings + position_embeddings
    res = self.llm_blocks(res)
    logits = self.lang_model_head(res) # format: B,T,vocab_size

    if targets is None:
      loss = None
    else:
      # batch, time, vocab_size
      B, T, C = logits.shape
      # Resize our tensors to match cross_entropy
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)

      # How well are we predicting the next character
      loss = FUN.cross_entropy(logits, targets)

    return logits, loss
  
  def generate(self, x, max_new_tokens):
    for _ in range(max_new_tokens):
      # Get the predictions (trim, bcs of BL_SIZE in pos_emb) (format B,T,C)
      logits, _ = self(x[:, -BLOCK_SIZE:])
      # Focus only on last time step (B,C)
      logits = logits[:, -1, :]
      # Get the probabilities
      probs = FUN.softmax(logits, dim=-1)
      # Sample from the prob distribution
      x_next = torch.multinomial(probs, num_samples=1) # format B,1
      # Append prediction to our generated sequence
      x = torch.cat((x, x_next), dim=1) # format B,T+1
    
    return x

if __name__ == '__main__':
  from helper.functions import loadData
  from helper.Trainer import Trainer
  from helper.tokenizer.Character import Character
  print('Loading data...', end='')

  data = loadData(r'C:\Users\ZabaJa\Desktop\AIs\input.txt')
  print('Done\n')

  print(f'Data: {len(data)} characters')

  tokenizer = Character(data)

  model = GPTTransformer(tokenizer.vocab_size).to(DEVICE)
  optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

  trainer = Trainer(model, optimizer, tokenizer, BLOCK_SIZE, BATCH_SIZE, STEPS, DEVICE)

  print('Start training')

  train_data = torch.tensor(tokenizer.tokenize(data), dtype=torch.long)

  trainer.train(train_data, print_progress=True)

  print(trainer.generate(400))
