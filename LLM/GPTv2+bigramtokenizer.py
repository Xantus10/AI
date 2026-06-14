import torch
from GPTv2 import GPTTransformer, BLOCK_SIZE, BATCH_SIZE, DEVICE

LEARNING_RATE = 3e-4
STEPS = 2000

if __name__ == '__main__':
  from helper.functions import loadData
  from helper.Trainer import Trainer
  from helper.tokenizer.FreqBigram import FreqBigram
  print('Loading data...', end='')

  data = loadData(r'C:\Users\ZabaJa\Desktop\AIs\input.txt')
  print('Done\n')

  print(f'Data: {len(data)} characters')

  tokenizer = FreqBigram(data, 90)

  model = GPTTransformer(tokenizer.vocab_size).to(DEVICE)
  optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

  trainer = Trainer(model, optimizer, tokenizer, BLOCK_SIZE, BATCH_SIZE, STEPS, DEVICE)

  print('Start training')

  train_data = torch.tensor(tokenizer.tokenize(data), dtype=torch.long)

  trainer.train(train_data, print_progress=True)

  print(trainer.generate(400))
