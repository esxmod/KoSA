from transformers import BertTokenizerFast

from modules.utils import escape_spec


class Tokenizer():
    def __init__(self, model, max_length):
        self.tokenizer = BertTokenizerFast.from_pretrained(
            model, cache_dir='cache/')
        self.max_length = max_length

    def tokenize(self, sentence):
        return self.tokenizer.encode_plus(text=escape_spec(sentence),
                                          padding='max_length',
                                          truncation=True,
                                          max_length=self.max_length)
