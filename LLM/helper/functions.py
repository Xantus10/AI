import torch


def loadData(filepath: str, size: int = -1):
  """
  Load the text data.

  Parameters
  ----------
  - filepath: The path to the file
  - size: How many characters to read
  """
  with open(filepath, 'r', encoding='utf-8') as f:
    # Content of training set
    return f.read(size)

def getBatch(data, block_size: int, batch_size: int):
  """
  Get a batch of data for the model

  Parameters
  ----------
  - data: The data (output from loadData* function)
  - block_size: Size of a single block of tokens
  - batch_size: Size of a single batch
  """
  ixs = torch.randint(len(data) - block_size, (batch_size,))
  base_data = torch.stack([data[i:i+block_size] for i in ixs])
  predict_data = torch.stack([data[i+1:i+block_size+1] for i in ixs])
  return base_data, predict_data
