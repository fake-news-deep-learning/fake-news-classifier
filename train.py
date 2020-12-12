from model import create_model, compile_model
from model_utils import create_callbacks


def train(x_train, y_train, x_valid, y_valid, vocab_size, embedding_dim, max_len):
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
    # set num filters to 36
    num_filters = 36

    # create CNN model
    cnn_model = create_model(num_filters, vocab_size, embedding_dim, max_len)
    # set metrics to use
    use_metrics = ['accuracy', 'TruePositives', 'TrueNegatives',
                   'FalsePositives', 'FalseNegatives']
    # compile model
    cnn_model = compile_model(cnn_model, use_metrics)

    # print model summary
    print(cnn_model.summary())

    # set callbacks to use
    use_callbacks = ['ModelCheckpoint', 'TensorBoard', 'EarlyStopping',
                     'ReduceLROnPlateau', 'TerminateOnNaN']

    history = cnn_model.fit(x_train, y_train,
                            epochs=10,
                            verbose=True,
                            callbacks=create_callbacks(use_callbacks),
                            validation_data=(x_valid, y_valid))

    return cnn_model, history
