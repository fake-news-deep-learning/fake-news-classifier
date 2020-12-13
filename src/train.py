import json

import numpy as np
from tqdm import tqdm

from model import create_model, compile_model
from model_utils import create_callbacks
from word2vec import prepare_tokenizer, text_to_sequence
from utils import load_glove, make_stratified_generator


def train(generator, valid, input_shape):
    """
        :param x_train:
        :param y_train:
        :param x_valid:
        :param y_valid:
        :param vocab_size:
        :param embedding_dim:
        :param max_len:

        :return:
    """

    steps = 38

    # set num filters to 36
    num_filters = 36

    # create CNN model
    cnn_model = create_model(num_filters, input_shape)
    # set metrics to use
    use_metrics = ['accuracy', 'TruePositives', 'TrueNegatives',
                   'FalsePositives', 'FalseNegatives']
    cnn_model = compile_model(cnn_model, use_metrics)
    print(cnn_model.summary())

    # set callbacks to use
    use_callbacks = ['ModelCheckpoint', 'TensorBoard', 'EarlyStopping',
                     'ReduceLROnPlateau', 'TerminateOnNaN']

    history = cnn_model.fit(
        generator,
        steps_per_epoch=steps,
        epochs=10,
        verbose=True,
        callbacks=create_callbacks(use_callbacks),
        validation_data=valid,
    )

    return cnn_model, history


def main():

    tokenizer = prepare_tokenizer()
    word2seq = load_glove(
        tokenizer.word_index,
        '../data/glove.6B/glove.6B.300d.txt'
    )

    print(f'Converting text from train set to sequences.')
    with open(f'../data/processed/train.json') as input_file:
        dataset = json.load(input_file)

    train_real = []
    train_fake = []

    for entry_id in tqdm(dataset):
        entry = dataset[entry_id]
        sequence = text_to_sequence(
            entry['text'],
            tokenizer.word_index,
            word2seq,
        )

        sequence = np.asarray(sequence, dtype=np.float32)

        if entry['label'] == 'real':
            train_real.append(sequence)
        else:
            train_fake.append(sequence)
    train_gen = make_stratified_generator(train_fake, train_real)

    print(f'Converting text from valid set to sequences.')
    with open(f'../data/processed/valid.json') as input_file:
        dataset = json.load(input_file)

    valid_x = []
    valid_y = []
    for entry_id in tqdm(dataset):
        entry = dataset[entry_id]
        sequence = text_to_sequence(
            entry['text'],
            tokenizer.word_index,
            word2seq,
        )

        valid_x.append(sequence)
        valid_y.append(0 if entry['label'] == 'fake' else 1)

    valid_y = np.asarray(valid_y, dtype=np.float32)
    valid_x = np.asarray(valid_x, dtype=np.float32)

    model = train(train_gen, (valid_x, valid_y), (70, 300, 1))


if __name__ == '__main__':
    main()
