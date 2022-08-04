import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration


class Summarization():
    def __init__(self):
        self.model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization', cache_dir='cache/')
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization', cache_dir='cache/')
        self.memo = {}

    def _predict(self, text):
        result = text.replace('\n', ' ')
        result = self.tokenizer.encode(result)
        result = [self.tokenizer.bos_token_id] + result + [self.tokenizer.eos_token_id]
        result = self.model.generate(torch.tensor([result]), num_beams=4, max_length=512, eos_token_id=1)
        result = self.tokenizer.decode(result.squeeze().tolist(), skip_special_tokens=True)
        result = [result]
        return result

    def predict(self, text):
        if text not in self.memo:
            self.memo[text] = self._predict(text)

        return self.memo[text]
