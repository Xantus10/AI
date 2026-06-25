import torch
from GPTv2 import GPTTransformer, BLOCK_SIZE, BATCH_SIZE, DEVICE, LEARNING_RATE

if __name__ == '__main__':
  from helper.functions import loadData
  from helper.Trainer import Trainer
  from helper.tokenizer.BPE import BPE
  print('Loading data...', end='')

  data = loadData(r'tinystories.txt', size=2000000)
  print('Done\n')

  print(f'Data: {len(data)} characters')

  tokenizer = BPE(data, special_tokens=['<|endoftext|>'], print_progress=True)

  model = GPTTransformer(tokenizer.vocab_size).to(DEVICE)
  optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

  trainer = Trainer(model, optimizer, tokenizer, BLOCK_SIZE, BATCH_SIZE, 8, DEVICE)

  print('Start training')

  data = torch.tensor(tokenizer.tokenize(data), dtype=torch.long)

  training_split = int(0.9 * len(data))
  train_data = data[:training_split]
  val_data = data[training_split:]

  trainer.train(train_data, val_data, print_progress=True)

  print(trainer.generate(400, start=torch.tensor([tokenizer.tokenize('<|endoftext|>\n')], dtype=torch.long)))
