import numpy as np

from transformers import BertTokenizerFast

from modules.utils import escape_spec


class Tokenizer():
    def __init__(self, model, max_length):
        self.tokenizer = BertTokenizerFast.from_pretrained(
            model, cache_dir='cache/')
        self.max_length = max_length

    def tokenize(self, sentence, to_tuple=False):
        encoded_dict = self.tokenizer.encode_plus(text=escape_spec(sentence),
                                                  padding='max_length',
                                                  truncation=True,
                                                  max_length=self.max_length)

        if not to_tuple:
            return encoded_dict

        input_ids = np.array(encoded_dict['input_ids']).reshape(1, -1)
        attention_mask = np.array(encoded_dict['attention_mask']).reshape(1, -1)
        token_type_ids = np.array(encoded_dict['token_type_ids']).reshape(1, -1)

        return (input_ids, attention_mask, token_type_ids)
