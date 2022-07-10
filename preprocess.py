import argparse
import numpy as np

from tqdm import tqdm

from modules.utils import load_dataset
from modules.tokenizer import Tokenizer


def preprocess(tokenizer, X, y):
    token_ids = []
    token_masks = []
    token_segments = []
    labels = []

    for idx in tqdm(range(len(X))):
        encoded_dict = tokenizer.tokenize(X.iloc[idx])

        token_ids.append(encoded_dict['input_ids'])
        token_masks.append(encoded_dict['attention_mask'])
        token_segments.append(encoded_dict['token_type_ids'])

        labels.append(y.iloc[idx])

    inputs = (
        np.array(token_ids),
        np.array(token_masks),
        np.array(token_segments)
    )
    labels = np.array(labels)

    return inputs, labels


def main(config):
    X_train, X_test, y_train, y_test = load_dataset(
        config.input_path, config.random_seed)

    tokenizer = Tokenizer(config.tokenizer, config.embed_dim)

    train = preprocess(tokenizer, X_train, y_train)
    test = preprocess(tokenizer, X_test, y_test)

    np.save(config.output_path, (train, test))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--tokenizer', type=str, default='kykim/bert-kor-base')
    parser.add_argument('--embed_dim', type=int, default=128)

    parser.add_argument('--random_seed', type=int, default=0xC0FFEE)

    parser.add_argument('--input_path', type=str, default='data/corpus.csv')
    parser.add_argument('--output_path', type=str,
                        default='data/corpus_encoded.npy')

    args = parser.parse_args()

    main(args)
