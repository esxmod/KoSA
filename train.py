import argparse
import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.metrics import accuracy_score

from models.kobert import TFKoBertModel


def main(config):
    train, test = np.load(config.input_path, allow_pickle=True)
    train_inputs, train_labels = train
    test_inputs, test_labels = test

    model = TFKoBertModel(config.model, config.input_dim, config.output_dim)

    optimizer = tfa.optimizers.RectifiedAdam(
        learning_rate=config.lr, weight_decay=0.0025, warmup_proportion=0.05)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=config.output_path,
                                                             monitor='val_sparse_categorical_accuracy',
                                                             save_best_only=True,
                                                             save_weights_only=True,
                                                             verbose=1)

    callback_earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy',
                                                          min_delta=0.0001,
                                                          patience=5)

    model.fit(train_inputs, train_labels, validation_split=0.2,
              epochs=config.num_epoch, batch_size=config.batch_size,
              verbose=1,
              callbacks=[callback_checkpoint, callback_earlystop])

    prediction = model.predict(test_inputs)
    prediction = tf.argmax(prediction, axis=1)
    score = accuracy_score(prediction, test_labels)

    print('test accuracy_score: ', score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='kykim/bert-kor-base')
    parser.add_argument('--input_dim', type=int, default=128)
    parser.add_argument('--output_dim', type=int, default=2)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epoch', type=int, default=5)

    parser.add_argument('--lr', type=float, default=1e-5)
    # parser.add_argument('--optim', type=str, default='Adam', choices=['SGD', 'Adam', 'RAdam'])
    # parser.add_argument('--dropout', type=float, default=0.2)

    parser.add_argument('--input_path', type=str,
                        default='data/corpus_encoded.npy')
    parser.add_argument('--output_path', type=str,
                        default='saved_models/best_weights.h5')

    args = parser.parse_args()

    main(args)
