import argparse
import numpy as np

from models.kobert import TFKoBertModel
from modules.tokenizer import Tokenizer


def preprocess(tokenizer, sentence):
    encoded_dict = tokenizer.tokenize(sentence)

    token_ids = np.array(encoded_dict['input_ids']).reshape(1, -1)
    token_masks = np.array(encoded_dict['attention_mask']).reshape(1, -1)
    token_segments = np.array(encoded_dict['token_type_ids']).reshape(1, -1)

    return (token_ids, token_masks, token_segments)


def main(config):
    tokenizer = Tokenizer(config.tokenizer, config.embed_dim)

    inputs = preprocess(tokenizer, config.input)

    model = TFKoBertModel(config.model, config.input_dim, config.output_dim)
    model.load_weights(filepath=config.checkpoint_path)

    prediction = model.predict(inputs)
    pred_probability = np.round(np.max(prediction) * 100, 2)
    pred_class = ['부정', '긍정'][np.argmax(prediction, axis=1)[0]]

    print(f'{pred_probability}% 확률로 {pred_class}입니다.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, default='감정을 분석할 문장입니다.')

    parser.add_argument('--tokenizer', type=str, default='kykim/bert-kor-base')
    parser.add_argument('--embed_dim', type=int, default=128)

    parser.add_argument('--model', type=str, default='kykim/bert-kor-base')
    parser.add_argument('--input_dim', type=int, default=128)
    parser.add_argument('--output_dim', type=int, default=2)
    parser.add_argument('--checkpoint_path', type=str,
                        default='saved_models/best_weights.h5')

    args = parser.parse_args()

    main(args)
