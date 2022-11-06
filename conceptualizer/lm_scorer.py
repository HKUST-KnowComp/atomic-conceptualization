# Used in atomic_parse.py to score different ways to add possessive pronoun or determiner using GPT2.

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class LMScorer():
    def __init__(self):
        self._lm = GPT2LMHeadModel.from_pretrained('gpt2')
        self._tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    def evaluate(self, text: str):
        tokens = torch.LongTensor(self._tokenizer.encode(text))
        loss = self._lm(input_ids=tokens, labels=tokens)[0]
        return (-loss).item()

if __name__ == '__main__':
    c = LMScorer()
    texts = ['Alice provides his framework.', 'Alice provides him framework.',
             'Alice provides him a framework.', "Alice provides him the framework."]
    for t in texts:
        print(c.evaluate(t))
