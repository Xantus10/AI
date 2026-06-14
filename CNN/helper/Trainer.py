import torch
from time import perf_counter

from .functions import getBatch

class Trainer:
  """
  Wrapper class for training and evaluation of Neural Network
  """
  def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, batch_size: int, epochs: int, device: torch.device | None = None):
    """
    Wrapper class for training and evaluation of Neural Network

    Parameters
    ----------
    - model: The model to train
    - optimizer: Optimizer to use
    - batch_size: Your configured batch size
    - epochs: How many training epochs to perform (Go over the entire dataset)
    - device (Optional): Device to use
    """
    self.model = model
    self.optimizer = optimizer
    self.batch_size = batch_size
    self.epochs = epochs
    self.device = device
  
  def _calculateSteps(self, dataset_size: int):
    """
    Calculate how many training steps to perform
    """
    return (dataset_size // self.batch_size) * self.epochs

  def train(self, train_data, print_progress: bool = False):
    """
    Train the Neural network

    Parameters
    ----------
    - train_data: The data to use for training
    - print_progress: Should the function print the training progress

    Return
    ------
    Time spent learning (in seconds)
    """
    self.model.train()
    start = perf_counter()
    steps = self._calculateSteps(len(train_data))
    if print_progress: print(f'Learning steps: {steps}')
    for step in range(steps):
      if print_progress and (step % (steps//100)) == 0: print(f'{step // (steps//100)}%')
      # Get training data
      labels, images = getBatch(train_data, self.batch_size)
      labels = labels.to(self.device)
      images = images.to(self.device)
      # Evaluate the loss
      _, loss = self.model(images, labels)
      # Zero out the gradients from the prev step
      self.optimizer.zero_grad(set_to_none=True)
      loss.backward()
      self.optimizer.step()
    end = perf_counter()

    if print_progress: print(f'Time spent learning: {round(end-start, 2)} s')
    return round(end-start, 2)

  def evaluate(self, test_data, print_progress: bool = False):
    """
    Evaluate the Neural network

    Parameters
    ----------
    - test_data: The data to use for evaluation
    - print_progress: Should the function print the evaluation progress

    Return
    ------
    The final accuracy of the model (in percent)
    """
    correct = 0
    processed = 0

    self.model.eval()

    with torch.no_grad():
      for i in range(len(test_data)//self.batch_size):
        labels, images = getBatch(test_data, self.batch_size)
        labels = labels.to(self.device)
        images = images.to(self.device)
        result, _ = self.model(images)
        predictions = torch.argmax(result, dim=-1)
        correct += (predictions == labels).sum().item()
        processed += self.batch_size
        if print_progress and i % 100 == 0: print(f'{round(correct/processed * 100, 2)}%')

    if print_progress: print(f'Final accuracy: {round(correct/processed * 100, 2)}% ({correct}/{processed})')
    return round(correct/processed * 100, 2)
