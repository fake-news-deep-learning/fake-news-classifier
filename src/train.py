import json
from typing import Tuple

import numpy as np
from tensorflow.keras import Model
from tqdm import tqdm

from model import create_text_cnn, compile_model
from model_utils import create_callbacks
from word2vec import prepare_tokenizer, text_to_sequence
from utils import load_glove, make_stratified_generator


def get_model(input_shape: Tuple) -> Model:
    """Instantiates a TextCNN and compiles it using a set of metrics."""
    cnn_model = create_text_cnn(input_shape)

    use_metrics = [
        "binary_accuracy",
        'TruePositives',
        'TrueNegatives',
        'FalsePositives',
        'FalseNegatives',
    ]

    cnn_model = compile_model(cnn_model, use_metrics)
    print(cnn_model.summary())

    return cnn_model


def get_lstm(input_shape: Tuple[int, int]) -> Model:
    """Instantiates a LSTM and compiles it using a set of metrics."""
    model = create_lstm_model(input_shape)

    use_metrics = [
        "binary_accuracy",
        'TruePositives',
        'TrueNegatives',
        'FalsePositives',
        'FalseNegatives',
    ]

    model = compile_model(model, use_metrics)
    print(model.summary())

    return model


def train(generator, steps, epochs, valid, input_shape) -> Tuple[Model, 'History']:
    """
    Args:
        generator: infinite generator for Keras format.
        steps: how many batches to use per epoch.
        epochs: how many epochs to train.
        valid: tuple of ndarrays (val_x, val_y)
        input_shape: 3-tuple with input matrix shape.

    Returns:
        The trained model and its training history.
    """
    cnn_model = get_model(input_shape)

    # set callbacks to use
    use_callbacks = [
        'ModelCheckpoint',
        'TensorBoard',
        'EarlyStopping',
        'ReduceLROnPlateau',
        'TerminateOnNaN',
    ]

    history = cnn_model.fit(
        generator,
        steps_per_epoch=steps,
        epochs=epochs,
        verbose=1,
        callbacks=create_callbacks(use_callbacks),
        validation_data=valid,
    )

    return cnn_model, history


def train_lstm(generator, steps, epochs, valid, input_shape) -> Tuple[Model, 'History']:
    """
    Args:
        generator: infinite generator for Keras format.
        steps: how many batches to use per epoch.
        epochs: how many epochs to train.
        valid: tuple of ndarrays (val_x, val_y)
        input_shape: 2-tuple with input matrix shape.

    Returns:
        The trained model and its training history.
    """
    model = get_lstm(input_shape)

    # set callbacks to use
    use_callbacks = [
        'ModelCheckpoint',
        'TensorBoard',
        'EarlyStopping',
        'ReduceLROnPlateau',
        'TerminateOnNaN',
    ]

    history = model.fit(
        generator,
        steps_per_epoch=steps,
        epochs=epochs,
        verbose=1,
        callbacks=create_callbacks(use_callbacks),
        validation_data=valid,
    )

    return model, history


def cnn_driver(glove: str, epochs: int = 20) -> Tuple[Model, 'History']:
    """
    Driver for training a TextCNN model. It performs:
        1. Fit Tokenizer on dataset.
        2. Convert train set words to sequences and fabricates a generator.
        3. Convert valid set words to sequences.
        4. Starts the training process.

    Args:
        glove: path to find glove file.

    Returns:
        The trained model and its training history.
    """
    tokenizer = prepare_tokenizer()
    word2seq = load_glove(tokenizer.word_index, glove)

    print(f'Converting text from train set to sequences.')
    with open(f'../data/processed/train.json') as input_file:
        dataset = json.load(input_file)

    train_real = []
    train_fake = []
    for entry in tqdm(dataset.values()):
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

    # preparing generator for training loop
    batch_size = 32
    steps = max(len(train_fake), len(train_real)) // (batch_size // 2)
    train_gen = make_stratified_generator(train_fake, train_real, batch_size)

    print(f'Converting text from valid set to sequences.')
    with open(f'../data/processed/valid.json') as input_file:
        dataset = json.load(input_file)

    valid_x = []
    valid_y = []
    for entry in tqdm(dataset.values()):
        sequence = text_to_sequence(
            entry['text'],
            tokenizer.word_index,
            word2seq,
        )
        valid_x.append(sequence)
        valid_y.append(0 if entry['label'] == 'fake' else 1)
    valid_y = np.asarray(valid_y, dtype=np.float32)
    valid_x = np.asarray(valid_x, dtype=np.float32)

    return train(train_gen, steps, epochs, (valid_x, valid_y), (70, 300, 1))


def lstm_driver(glove: str, epochs: int = 20) -> Tuple[Model, 'History']:
    """
    Driver for training a LSTM model. It performs:
        1. Fit Tokenizer on dataset.
        2. Convert train set words to sequences and fabricates a generator.
        3. Convert valid set words to sequences.
        4. Starts the training process.

    Args:
        glove: path to find glove file.

    Returns:
        The trained model and its training history.
    """
    tokenizer = prepare_tokenizer()
    word2seq = load_glove(tokenizer.word_index, glove)

    print(f'Converting text from train set to sequences.')
    with open(f'../data/processed/train.json') as input_file:
        dataset = json.load(input_file)

    train_real = []
    train_fake = []
    for entry in tqdm(dataset.values()):
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

    # preparing generator for training loop
    batch_size = 32
    steps = max(len(train_fake), len(train_real)) // (batch_size // 2)
    train_gen = make_stratified_generator(train_fake, train_real, batch_size)

    print(f'Converting text from valid set to sequences.')
    with open(f'../data/processed/valid.json') as input_file:
        dataset = json.load(input_file)

    valid_x = []
    valid_y = []
    for entry in tqdm(dataset.values()):
        sequence = text_to_sequence(
            entry['text'],
            tokenizer.word_index,
            word2seq,
            mode='lstm'
        )
        valid_x.append(sequence)
        valid_y.append(0 if entry['label'] == 'fake' else 1)
    valid_y = np.asarray(valid_y, dtype=np.float32)
    valid_x = np.asarray(valid_x, dtype=np.float32)

    return train_lstm(train_gen, steps, epochs, (valid_x, valid_y), (70, 300))
