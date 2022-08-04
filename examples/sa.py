from models.kobert import TFKoBertModel
from modules.tokenizer import Tokenizer


class SentimentAnalysis():
    def __init__(self):
        self.model = TFKoBertModel('kykim/bert-kor-base', 128, 2)
        self.tokenizer = Tokenizer('kykim/bert-kor-base', 128)
        self.memo = {}

        self.model.load_weights('saved_models/best_weights.h5')

    def _predict(self, text):
        result = self.tokenizer.tokenize(text, True)
        result = self.model.predict(result)
        result = result.tolist()[0]
        return result

    def predict(self, text):
        if text not in self.memo:
            self.memo[text] = self._predict(text)

        return self.memo[text]
