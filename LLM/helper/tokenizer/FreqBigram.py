from collections import defaultdict
from heapq import nlargest

from .Tokenizer import Tokenizer

class FreqBigram(Tokenizer):
  """
  Tokenizer using the most frequent bigrams in the text + characters
  """
  def __init__(self, text: str, bigrams: int = 90):
    """
    Tokenizer using the most frequent bigrams in the text + characters

    Parameters
    ----------
    - text: The text which will be tokenized
    - bigrams: How many most frequent bigrams to take into account
    """
    self.chars = sorted(list(set(text)))
    self.no_bigrams = bigrams
    self._addBigramsToChars(text)
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
    ret = []
    i = 0
    ln = len(s)
    chk_last = False
    if ln % 2 == 1:
      ln -= 1
      chk_last = True
    while i < ln:
      c = s[i:i+2]
      if c in self.stoi.keys():
        ret.append(self.stoi[c])
        i += 1
      else:
        ret.append(self.stoi[c[0]])
      i += 1
    if chk_last:
      ret.append(self.stoi[s[-1]])

    return ret
  
  def detokenize(self, l):
    return ''.join([self.itos[i] for i in l])
