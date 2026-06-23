from .Tokenizer import Tokenizer

import re
from collections import defaultdict
from heapq import nlargest

class BPE(Tokenizer):
  """
  Tokenizer using the Byte Pair Encoding (Primitive, Not optimized)
  """
  def __init__(self, text: str, target_vocab_size: int = 300, special_tokens: list[str] = [], print_progress: bool=False):
    """
    Tokenizer using the Byte Pair Encoding

    Parameters
    ----------
    - text: The text which will be tokenized
    - target_vocab_size: Continue with the BPE until this threshold
    - special_tokens: Special character sequences that should always be treated as separate tokens
    - print_progress: Print the BPE algorithm progress
    """
    self.chars = sorted(list(set(text)))
    self.target_vocab_size = target_vocab_size
    self.target_vocab_size += len(special_tokens)

    self.print_progress = print_progress

    self.special_tokens = special_tokens
    self.special_regex = re.compile(f'({"|".join([re.escape(s) for s in special_tokens])})') if len(special_tokens) > 0 else re.compile('(?!)')
    self.chars += special_tokens

    super().__init__(len(self.chars),
                     {c: i for i, c in enumerate(self.chars)},
                     {i: c for i, c in enumerate(self.chars)})
    self._runBPE(text)

  def _runBPE(self, text: str):
    tokens = self._toTokens(text)
    while self.vocab_size < self.target_vocab_size:
      mergers = self._findMergeTokens(tokens)
      self._mergeBytePairs(tokens, mergers)
      if self.print_progress: print(f'Vocab size: {self.vocab_size} of {self.target_vocab_size}')

  def _toTokens(self, s: str):
    return [self.stoi[c] for c in s]

  def _findMergeTokens(self, tokens: list[int]):
    bigrams_freq = defaultdict(int)

    for i in range(len(tokens)-1):
      bigrams_freq[(tokens[i], tokens[i+1])] += 1
    
    n = 1 if self.vocab_size > int(self.target_vocab_size * 0.7) else max(1, (self.target_vocab_size - self.vocab_size) // 10)

    largest = nlargest(n, bigrams_freq, bigrams_freq.get)

    if n == 1: return largest

    uniq = []
    ret = []

    for t in largest:
      if t[0] in uniq or t[1] in uniq:
        continue
      uniq.append(t[0])
      uniq.append(t[1])
      ret.append(t)
    
    return ret
  
  def _mergeBytePairs(self, tokens: list[int], mergers: tuple[int, int]):
    new_token_ids = [self._addToken(t) for t in mergers]
    ln = len(tokens)-1
    i = 0
    while i < ln:
      if (tokens[i], tokens[i+1]) in mergers:
        tokens[i:i+2] = [new_token_ids[mergers.index((tokens[i], tokens[i+1]))]]
        ln -= 1
      i += 1

  def _addToken(self, token: tuple[int, int]):
    self.stoi[token] = self.vocab_size
    self.itos[self.vocab_size] = token
    self.vocab_size += 1
    return self.vocab_size-1

  def tokenize(self, s):
    ret = re.split(self.special_regex, s)
    return [token for item in ret for token in self._subtokenize(item)]

  def _subtokenize(self, s):
    if s in self.special_tokens:
      return [self.stoi[s]]
    tokens = self._toTokens(s)
    keys = list(self.stoi.keys())
    changed = True
    while changed:
      changed = False
      i = 0
      ln = len(tokens)-1
      while i < ln:
        pair = (tokens[i], tokens[i+1])
        if pair in keys:
          tokens[i:i+2] = [self.stoi[pair]]
          changed = True
          ln -= 1
        i += 1
    return tokens

  def detokenize(self, l):
    compound = True
    while compound:
      compound = False
      i = 0
      while i < len(l):
        if not isinstance(l[i], str):
          sub = self.itos[l[i]]
          if isinstance(sub, tuple): compound = True
          l[i:i+1] = sub
        i += 1
    return ''.join(l)
