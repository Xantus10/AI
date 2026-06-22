from abc import ABC, abstractmethod
from typing import Any

class Tokenizer(ABC):
  """
  Template class for all tokenizers
  """
  def __init__(self, vocab_size: int, stoi: dict[Any, int], itos: dict[int, Any]):
    """
    Template class for all tokenizers

    Parameters
    ----------
    - vocab_size: Size of the vocabulary
    - stoi: A mapping of string tokens to int values
    - itos: A reverse mapping fot stoi
    """
    self.vocab_size = vocab_size
    self.stoi = stoi
    self.itos = itos
  
  @abstractmethod
  def tokenize(self, s: str) -> list[int]:
    """
    Turn a string into a list of tokens
    """
    pass

  @abstractmethod
  def detokenize(self, l: list[int]) -> str:
    """
    Turn a list of tokens into a string
    """
    pass
