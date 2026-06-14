import torch
import torch.nn as nn
from torch.nn import functional as FUN

### Constants
# Channels of the image
IMG_CHANNELS = 1 # Greyscale
# Dimensions of the images
IMG_DIMENSIONS = 28
# Output numbers
OUTPUT_NUMBERS = 10
# Batch size
BATCH_SIZE = 4
# How many base features should the network recognize
BASE_FEATURES = 20
# How many base features should the network recognize
HIDDEN_LAYER = 160
# Learning steps
STEPS = 800
# Learning rate
LEARNING_RATE = 3e-4
# Enable running on GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# Dropout rate
DROPOUT = 0.2


class ImageProcessingLayer(nn.Module):
  """
  A single Convolution layer
  """
  def __init__(self, layer_ix: int = 0):
    """
    A single Convolution layer

    Parameters
    ----------
    - layer_ix: The index of this layer
    """
    super().__init__()
    inp_size = IMG_CHANNELS if layer_ix == 0 else BASE_FEATURES * layer_ix
    out_size = BASE_FEATURES * (layer_ix+1)
    self.layer = nn.Sequential(
      # The convolution - Learnable matrix will slide accross the image and pick up edges/shapes
      nn.Conv2d(inp_size, out_size, kernel_size=3, padding=1, bias=False),
      # Normalize the (potentially large) values to be smaller and around 0
      nn.BatchNorm2d(out_size),
      # Non-linearity, -n => 0
      nn.ReLU(inplace=True),
      # Select the max value in the kernel -> Will shrink the image H/W by 1/2 (75% data)
      nn.MaxPool2d(kernel_size=2, stride=2)
    )
  
  def forward(self, x):
    x = self.layer(x) # X will be B,C,W,H
    return x

class ImageRecognitionModel(nn.Module):
  """
  Image recognition model
  """
  def __init__(self, image_processing_layers: int = 2):
    """
    Image recognition model

    Parameters
    ----------
    - image_processing_layers: Number of convolution layers
    """
    super().__init__()
    # Image processing layers - Extract shapes/edges/textures
    self.img_processing = nn.Sequential(
      *[ImageProcessingLayer(i) for i in range(image_processing_layers)]
    )
    # Flatten the 2D image into a 1D array
    self.flatten = nn.Flatten()
    # First linear layer of the network, will output the visual features
    self.hidden_layer = nn.Linear(BASE_FEATURES * image_processing_layers * ((IMG_DIMENSIONS // (2*image_processing_layers))**2), HIDDEN_LAYER)
    # Non-linearity between the Linear layers
    self.relu = nn.ReLU(inplace=True)
    # Avoid overfitting
    self.dropout = nn.Dropout(DROPOUT)
    # Final mapping layer, will output the result of the prediction
    self.final_layer = nn.Linear(HIDDEN_LAYER, OUTPUT_NUMBERS)
  
  def forward(self, x, target = None):
    x = self.img_processing(x)
    x = self.flatten(x)
    x = self.hidden_layer(x)
    res = self.final_layer(x) # dimensions: B,10 | target: Array of size B

    if target is None:
      loss = None
    else:
      # How well are we predicting the next character
      loss = FUN.cross_entropy(res, target)

    return res, loss

if __name__ == '__main__':
  from helper.functions import loadDataCSV
  from helper.Trainer import Trainer
  print('Loading data...', end='')

  train_data = loadDataCSV(r'MNIST\mnist_train.csv', image_channels=IMG_CHANNELS, image_dimenisons=(IMG_DIMENSIONS, IMG_DIMENSIONS))
  test_data = loadDataCSV(r'MNIST\mnist_test.csv', image_channels=IMG_CHANNELS, image_dimenisons=(IMG_DIMENSIONS, IMG_DIMENSIONS))

  print('Done\n')

  print(f'Train data: {len(train_data)} images')
  print(f'Test data: {len(test_data)} images')

  model = ImageRecognitionModel().to(DEVICE)
  optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

  trainer = Trainer(model, optimizer, BATCH_SIZE, STEPS, DEVICE)

  print('Start training')

  trainer.train(train_data, print_progress=True)

  trainer.evaluate(test_data, print_progress=True)

