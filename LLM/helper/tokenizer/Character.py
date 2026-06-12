from .Tokenizer import Tokenizer

class Character(Tokenizer):
  """
  A single character tokenizer
  """
  def __init__(self, text: str):
    """
    A single character tokenizer

    Parameters
    ----------
    - text: The text which will be tokenized
    """
    self.chars = sorted(list(set(text)))
    super().__init__(len(self.chars),
                     {c: i for i, c in enumerate(self.chars)},
                     {i: c for i, c in enumerate(self.chars)})
  
  def tokenize(self, s):
    return [self.stoi[c] for c in s]
  
  def detokenize(self, l):
    return ''.join([self.itos[i] for i in l])
