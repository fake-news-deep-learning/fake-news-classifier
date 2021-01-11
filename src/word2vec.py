"""Converts the dataset to sequences using word2vec."""
import json

import numpy as np
from numpy import ndarray
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer

from utils import load_glove

# NLTK stopwords
stopwords = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
    "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it',
    "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
    'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is',
    'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
    'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',
    've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
    'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
    'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
    "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
    "weren't", 'won', "won't", 'wouldn', "wouldn't"
]


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
    words = sorted(
        list(tkn.word_counts.items()),
        key=lambda x: x[1],
    )
    print(f'Corpus has {len(words)} words.')

    # remove rare words
    rare_words = [word for word, cnt in tkn.word_counts.items() if cnt < 11]
    for word in rare_words:
        del tkn.word_index[word]
        del tkn.word_docs[word]
        del tkn.word_counts[word]

    rare_words_percentage = 100*len(rare_words)/len(words)
    print(
        f'Removed rare words, which were {rare_words_percentage:.2f}% of corpus.'
    )

    # remove stopwords
    cnt = 0
    for word in stopwords:
        try:
            del tkn.word_index[word]
            del tkn.word_docs[word]
            del tkn.word_counts[word]
            cnt += 1
        except KeyError:
            pass
    print(f'Succesfully removed {cnt} stopwords.')

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
