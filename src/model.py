import tensorflow
from tensorflow.python import keras

from keras.models import Sequential
from keras import layers
from model_utils import create_metrics


def create_model(num_filters, input_shape):
    """
        :param num_filters:
        :param vocab_size:
        :param embedding_dim:
        :param maxlen:

        :return:
    """
    embed_size = input_shape[1]
    model = Sequential()
    model.add(layers.InputLayer(input_shape))
    for kernel_size in [1, 2, 3, 5]:
        model.add(layers.Conv2D(
            num_filters,
            (kernel_size, embed_size),
            padding='same',
            name=f'conv2d_kernel_{kernel_size}_layer'),
        )
    model.add(layers.Dropout(0.1, name='dropout_layer'))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, name='linear_layer', activation='sigmoid'))

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
