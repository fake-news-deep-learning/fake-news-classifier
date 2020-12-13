"""Converts the dataset to sequences using word2vec."""
import json

import numpy as np
from numpy import ndarray
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer

from utils import load_glove


def prepare_tokenizer() -> Tokenizer:
    """Fits a Keras Tokenizer on the dataset."""
    tkn = Tokenizer()
    dataset = {}

    for filename in ['train', 'test', 'valid']:
        with open(f'../data/processed/{filename}.json') as input_file:
            dataset.update(json.load(input_file))

    print('Fitting Tokenizer to dataset.')
    tkn.fit_on_texts([entry['text'] for entry in tqdm(dataset.values())])

    # get words sorted by counts and remove the most commons
    counts = sorted(
        list(tkn.word_counts.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    remove_top = 42
    for i in range(1, remove_top + 1):
        word = counts[i][0]
        print(f'removing {word} from vocab')
        del tkn.word_index[word]
        del tkn.word_docs[word]
        del tkn.word_counts[word]

    # remove rare words
    rare_words = [word for word, cnt in tkn.word_counts.items() if cnt < 11]
    for word in rare_words:
        del tkn.word_index[word]
        del tkn.word_docs[word]
        del tkn.word_counts[word]

    return tkn


def text_to_sequence(text, word_index, word2seq, length=70) -> ndarray:
    """Converts a text to its matrix (embed_size x length) representation."""

    embed_size = len(word2seq[0])
    sequence = []

    for word in text.split():

        if word in word_index:
            sequence.append(word2seq[word_index[word]])

        # trims long texts
        if len(sequence) == length:
            break

    # pads short sequences
    while len(sequence) < length:
        sequence.append(np.zeros(embed_size))

    return np.asarray(sequence).reshape(length, embed_size, 1)


# def main():
#     tokenizer = prepare_tokenizer()
#     word2seq = load_glove(
#         tokenizer.word_index,
#         '../data/glove.6B/glove.6B.300d.txt'
#     )
#
#     for filename in ['train', 'test', 'valid']:
#
#         with open(f'../data/processed/{filename}.json') as input_file:
#             dataset = json.load(input_file)
#
#         print(f'Converting text from set {filename} to sequences.')
#         for entry_id in tqdm(dataset):
#             entry = dataset[entry_id]
#             sequence = text_to_sequence(
#                 entry['text'], tokenizer.word_index, word2seq)
#             dataset[entry_id]['sequence'] = sequence
#             del dataset[entry_id]['text']
#
#         with open(f'../data/interim/{filename}.json', 'w') as out_file:
#             json.dump(dataset, out_file, sort_keys=True, indent=2)
