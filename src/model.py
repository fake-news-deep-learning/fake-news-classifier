from typing import List, Tuple

from tensorflow.python import keras
from keras import Model, layers
from keras.models import Sequential

from model_utils import create_metrics


def create_text_cnn(input_shape: Tuple) -> Model:
    """Implements the TextCNN architecture using given input shape."""
    filters_per_layer = 36
    embed_size = input_shape[1]

    model = Sequential()
    model.add(layers.InputLayer(input_shape))

    for kernel_size in [1, 2, 3, 5]:
        model.add(layers.Conv2D(
            filters_per_layer,
            (kernel_size, embed_size),
            padding='same',
            name=f'conv2d_kernel_{kernel_size}_layer'),
        )

    model.add(layers.Dropout(0.1, name='dropout_layer'))
    model.add(layers.Flatten())
    model.add(layers.Dense(output_dim=1, name='linear_layer', activation='sigmoid'))

    return model


def create_lstm_model(input_shape: Tuple) -> Model:
    """
        :param num_filters:
        :param input_shape:

        :return:
    """
    # embedding_dim = input_shape[1]
    print(input_shape)

    model = Sequential()

    # model.add(layers.InputLayer(input_shape=(70, )))

    model.add(layers.Bidirectional(layers.LSTM(256,  
                                               return_sequences=False))
                                    input_shape=(300, 70))

    # model.add(layers.Flatten())
    model.add(layers.Dense(1,
                           activation='sigmoid',
                           name='linear_layer'))

    return model


def compile_model(model_to_compile: Model, metrics_names: List[str]) -> Model:
    """Compiles a Model using given metrics names."""

    model_to_compile.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=create_metrics(metrics_names),
    )

    return model_to_compile
