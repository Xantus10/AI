import re
from collections import defaultdict
from heapq import nlargest

from .Tokenizer import Tokenizer

class FreqBigram(Tokenizer):
  """
  Tokenizer using the most frequent bigrams in the text + characters
  """
  def __init__(self, text: str, bigrams: int = 90, special_tokens: list[str] = []):
    """
    Tokenizer using the most frequent bigrams in the text + characters

    Parameters
    ----------
    - text: The text which will be tokenized
    - bigrams: How many most frequent bigrams to take into account
    - special_tokens: Special character sequences that should always be treated as separate tokens
    """
    self.chars = sorted(list(set(text)))
    self.no_bigrams = bigrams
    self._addBigramsToChars(text)

    self.special_tokens = special_tokens
    self.special_regex = re.compile(f'({"|".join([re.escape(s) for s in special_tokens])})')
    self.chars += special_tokens

    super().__init__(len(self.chars),
                     {c: i for i, c in enumerate(self.chars)},
                     {i: c for i, c in enumerate(self.chars)})
  
  def _addBigramsToChars(self, text: str):
    bigrams_freq = defaultdict(int)

    for i in range(len(text)-1):
      bigrams_freq[text[i:i+2]] += 1

    frequent_bigrams = nlargest(self.no_bigrams, bigrams_freq, key=bigrams_freq.get)

    self.chars += frequent_bigrams

  def tokenize(self, s):
    ret = re.split(self.special_regex, s)
    return [token for item in ret for token in self._subtokenize(item)]

  def _subtokenize(self, s):
    if s in self.special_tokens:
      return [self.stoi[s]]
    ret = []
    i = 0
    ln = len(s) - 1
    while i <= ln:
      if i == ln:
        ret.append(self.stoi[s[-1]])
      else:
        c = s[i:i+2]
        if c in self.stoi.keys():
          ret.append(self.stoi[c])
          i += 1
        else:
          ret.append(self.stoi[c[0]])
      i += 1

    return ret
  
  def detokenize(self, l):
    return ''.join([self.itos[i] for i in l])
