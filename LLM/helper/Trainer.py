import torch
from time import perf_counter

from .functions import getBatch
from .tokenizer.Tokenizer import Tokenizer

class Trainer:
  """
  Wrapper class for training and evaluation of Neural Network
  """
  def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, tokenizer: Tokenizer, block_size: int, batch_size: int, epochs: int, device: torch.device | None = None):
    """
    Wrapper class for training and evaluation of Neural Network

    Parameters
    ----------
    - model: The model to train
    - optimizer: Optimizer to use
    - tokenizer: Used tokenizer
    - block_size: Your configured block size
    - batch_size: Your configured batch size
    - epochs: How many training epochs to perform
    - device (Optional): Device to use
    """
    self.model = model
    self.optimizer = optimizer
    self.tokenizer = tokenizer
    self.block_size = block_size
    self.batch_size = batch_size
    self.epochs = epochs
    self.device = device
  
  def _calculateSteps(self, dataset_size: int):
    """
    Calculate how many training steps to perform
    """
    return (dataset_size // (self.batch_size * self.block_size))
  
  def train(self, train_data, validation_data = None, print_progress: bool = False):
    """
    Train the Neural network

    Parameters
    ----------
    - train_data: The data to use for training
    - validation_data (optional): The data to use for validation after each epoch
    - print_progress: Should the function print the training progress

    Return
    ------
    Time spent learning (in seconds)
    """
    start = perf_counter()
    steps = self._calculateSteps(len(train_data))
    total_loss = 0.0
    if not validation_data is None:
      val_steps = self._calculateSteps(len(validation_data))
    if print_progress:
      print(f'Epochs / Steps per epoch: {self.epochs} / {steps}')
      print('Epoch - Training loss - Validation loss')
    for e in range(self.epochs):
      self.model.train()
      for step in range(steps):
        # Get training data
        base_data, predict_data = getBatch(train_data, self.block_size, self.batch_size)
        predict_data = predict_data.to(self.device)
        base_data = base_data.to(self.device)
        # Evaluate the loss
        _, loss = self.model(base_data, predict_data)
        # Zero out the gradients from the prev step
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        total_loss += loss.item()
      val_loss = 0.0
      if not validation_data is None:
        self.model.eval()
        with torch.no_grad():
          for _ in range(val_steps):
            base_data, predict_data = getBatch(validation_data, self.block_size, self.batch_size)
            predict_data = predict_data.to(self.device)
            base_data = base_data.to(self.device)
            _, loss = self.model(base_data, predict_data)
            val_loss += loss.item()
      if print_progress:
        print(f'{e+1:<5} - {round(total_loss / steps, 3):^13} - {round(val_loss / val_steps, 3) if not validation_data is None else '-':^15}')
      total_loss = 0.0
    end = perf_counter()

    if print_progress: print(f'Time spent learning: {round(end-start, 2)} s')
    return round(end-start, 2)

  def generate(self, count: int = 100, start = torch.zeros((1, 1), dtype=torch.long)):
    """
    Evaluate the Neural network

    Parameters
    ----------
    - count: How many characters to generate

    Return
    ------
    The generated text
    """
    self.model.eval()

    with torch.no_grad():
      generated = self.model.generate(start, max_new_tokens=count)
      return self.tokenizer.detokenize(generated[0].tolist())
