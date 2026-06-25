from .Tokenizer import Tokenizer
from typing import Any
import re

class TokenizerSpecialTokens(Tokenizer):
  """
  Template class for all tokenizers with special_tokens property
  """
  def __init__(self, vocab_size: int, stoi: dict[Any, int], itos: dict[int, Any], special_tokens: list[str] = []):
    """
    Template class for all tokenizers

    Parameters
    ----------
    - vocab_size: Size of the vocabulary
    - stoi: A mapping of string tokens to int values
    - itos: A reverse mapping fot stoi
    - special_tokens: Special character sequences that should always be treated as separate tokens
    """
    super().__init__(vocab_size + len(special_tokens),
                      stoi | {c: (vocab_size+i) for i, c in enumerate(special_tokens)},
                      itos | {(vocab_size+i): c for i, c in enumerate(special_tokens)})
    self.special_tokens = special_tokens
    self.special_regex = re.compile(f'({"|".join([re.escape(s) for s in special_tokens])})') if len(special_tokens) > 0 else re.compile('(?!)')
  
  def splitSpecialTokens(self, data: str):
    return re.split(self.special_regex, data)
