from typing import List, Tuple

from tensorflow.python import keras
from keras import Model, layers, backend
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

    filters_per_layer = 36
    embed_size = input_shape[1]

    model = Sequential()

    model.add(layers.InputLayer(input_shape))

    for kernel_size in [1, 2, 3, 5]:
        model.add(layers.Conv2D(
            filters_per_layer,
            (kernel_size, embed_size),
            padding='same',
            name=f'conv2d_kernel_{kernel_size}_layer'),)

    # model.add(layers.LSTM(64,
    #                       batch_input_size=(None, input_shape[1], input_shape[2]),
    #                       return_sequences=True))

    squeezed = Lambda(lambda x: backend.squeeze(x, 2))(model_in)
    # LSTM(10)(squeezed)
    model.add(layers.LSTM(100)(squeezed))

    # model.add(layers.Dense(64))

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(1, activation='sigmoid'))

    return model


def compile_model(model_to_compile: Model, metrics_names: List[str]) -> Model:
    """Compiles a Model using given metrics names."""

    model_to_compile.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=create_metrics(metrics_names),
    )

    return model_to_compile
