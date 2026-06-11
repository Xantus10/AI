import torch
import pandas as pd


def loadDataCSV(filepath: str, image_channels: int, image_dimenisons: tuple[int, int]):
  """
  Load the image data from CSV

  Parameters
  ----------
  - filepath: The path to the CSV
  - image_channels: Channels of the image (1=grayscale, 3=RGB)
  - image_dimensions: Tuple of width/height
  """
  raw = pd.read_csv(filepath, header=None)
  labels = raw.iloc[:, 0].tolist()
  all_pixels = raw.iloc[:, 1:].values.astype('long')

  all_tensors = (torch.tensor(all_pixels, dtype=torch.float32) / 255.0).reshape(-1, image_channels, image_dimenisons[0], image_dimenisons[1])

  return list(zip(labels, all_tensors))


def getBatch(data, batch_size: int):
  """
  Get a batch of data for the model

  Parameters
  ----------
  - data: The data (output from loadData* function)
  - batch_size: Size of a single batch
  """
  ixs = torch.randint(len(data), (batch_size,))
  labels = torch.tensor([data[i][0] for i in ixs], dtype=torch.long)
  tensors = torch.stack([data[i][1] for i in ixs])
  return labels, tensors
