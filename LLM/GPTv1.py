import torch
import torch.nn as nn
from torch.nn import functional as FUN

### Constants
# Size of training blocks
BLOCK_SIZE = 8
# Paralel processing (efficiency)
BATCH_SIZE = 32
# Learning epochs
EPOCHS = 10
# Learning rate
LEARNING_RATE = 1e-3
# Enable running on GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class BigramLangModel(nn.Module):
  """
  The simplest GPT model
  """
  def __init__(self, vocab_size: int):
    """
    The simplest GPT model

    Parameters
    ----------
    - vocab_size: Size of the vocabulary
    """
    super().__init__()
    # A single embedding (Table mapping token -> list of probabilities of following tokens)
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
  
  def forward(self, x, targets=None):
    # Predicting what comes next based just on x (format: B,T,C)
    logits = self.token_embedding_table(x)

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
      # Get the predictions (format B,T,C)
      logits, _ = self(x)
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

  data = loadData(r'input.txt')
  print('Done\n')

  print(f'Data: {len(data)} characters')

  tokenizer = Character(data)

  model = BigramLangModel(tokenizer.vocab_size).to(DEVICE)
  optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

  trainer = Trainer(model, optimizer, tokenizer, BLOCK_SIZE, BATCH_SIZE, EPOCHS, DEVICE)

  print('Start training')

  train_data = torch.tensor(tokenizer.tokenize(data), dtype=torch.long)

  trainer.train(train_data, print_progress=True)

  print(trainer.generate(400))
