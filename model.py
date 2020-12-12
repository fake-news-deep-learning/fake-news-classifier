from keras.models import Sequential
from keras import layers
from model_utils import create_metrics


def create_model(num_filters, vocab_size, embedding_dim, maxlen, filters_size=[1,2,3,5]):
    """
    """
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen,
                               name='embedding_layer'))

    for kernel_size in filters_size:
        model.add(layers.Conv1D(num_filters, kernel_size,
                                name=f'conv1d_kernel_{kernel_size}_layer'))

    model.add(layers.Dropout(0.1, name='dropout_layer'))
    model.add(layers.Dense(1, name='linear_layer'))

    return model


def compile_model(model_to_compile, metrics_names=['accuracy']):
    model_to_compile.compile(optimizer='adam',
                             loss='binary_crossentropy',
                             metrics=create_metrics(metrics_names))

    return model


if __name__ == '__main__':
    model = create_model(num_filters=36, vocab_size=5000, embedding_dim=50, maxlen=100)

    use_metrics = ['accuracy', 'TruePositives', 'TrueNegatives',
                   'FalsePositives', 'FalseNegatives']
    model = compile_model(model, use_metrics)

    print(model.summary())
