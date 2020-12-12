from keras.models import Sequential
from keras import layers
from model_utils import create_metrics


def create_model(num_filters, vocab_size, embedding_dim, maxlen):
    """
        :param num_filters:
        :param vocab_size:
        :param embedding_dim:
        :param maxlen:

        :return:
    """
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen,
                               name='embedding_layer'))

    for kernel_size in [1, 2, 3, 5]:
        model.add(layers.Conv1D(num_filters, kernel_size,
                                name=f'conv1d_kernel_{kernel_size}_layer'))

    model.add(layers.Dropout(0.1, name='dropout_layer'))
    model.add(layers.Dense(1, name='linear_layer'))

    return model


def compile_model(model_to_compile, metrics_names):
    """
        :param model_to_compile:
        :param metrics_names:

        :return:
    """
    model_to_compile.compile(optimizer='adam',
                             loss='binary_crossentropy',
                             metrics=create_metrics(metrics_names))

    return model_to_compile


if __name__ == '__main__':
    test_model = create_model(num_filters=36, vocab_size=5000, embedding_dim=50, maxlen=100)

    use_metrics = ['accuracy', 'TruePositives', 'TrueNegatives',
                   'FalsePositives', 'FalseNegatives']
    test_model = compile_model(test_model, use_metrics)

    print(test_model.summary())
