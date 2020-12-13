import numpy as np
from itertools import cycle
from tqdm import tqdm


def make_stratified_generator(fake, real, batch_size=16):
    """
    Use iterables to fabricate an infinite generator to use in Keras.
    Each batch is balanced.
    """
    iter_first = cycle(fake)
    iter_second = cycle(real)

    batch_size = batch_size // 2
    batch_y = [0, 1] * batch_size
    batch_y = np.asarray(batch_y, dtype=np.float32)

    while True:
        batch_x = []
        for i in range(batch_size):
            batch_x.append(next(iter_first))
            batch_x.append(next(iter_second))

        yield np.asarray(batch_x, dtype=np.float32), batch_y)


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype = 'float32')


def load_glove(word_index, path):
    print(f'Mapping vocabulary with size {len(word_index)} to sequences')

    max=0
    for index in word_index.values():
        if index > max:
            max=index

    # parsing GLOVE
    print(f'Parsing GLOVE')
    embeddings_index=dict(get_coefs(*line.split(" "))
                            for line in tqdm(open(path)))
    embed_example=next(iter(embeddings_index.values()))
    embed_size=len(embed_example)
    print(f'word2vec embeds have len {embed_size}')

    # random initialization for the word2vec matrix
    emb_mean, emb_std=-0.005838499, 0.48782197
    embedding_matrix=np.random.normal(
        emb_mean, emb_std, (max, embed_size)
    )

    not_found=0
    print(f'Translating vocabulary')
    for word, i in tqdm(word_index.items()):

        # get vec for current `word`, fallbacks to `WORD`
        embedding_vector=embeddings_index.get(word)

        if embedding_vector is not None:
            embedding_matrix[i]=embedding_vector
        else:
            embedding_vector=embeddings_index.get(word.capitalize())
            if embedding_vector is not None:
                embedding_matrix[i]=embedding_vector
            else:
                not_found += 1

    print(f'Could not find {not_found} words in GLOVE.')

    return embedding_matrix
