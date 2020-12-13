import os
from keras import metrics, callbacks


metrics_dict = {
    'accuracy': metrics.Accuracy(name='accuracy'),
    'binary_accuracy': metrics.BinaryAccuracy(name='binary_accuracy'),
    'categorical_accuracy': metrics.CategoricalAccuracy(name='categorical_accuracy'),
    'AUC': metrics.AUC(name='AUC'),
    'precision': metrics.Precision(),
    'recall': metrics.Recall(),
    'TruePositives': metrics.TruePositives(),
    'TrueNegatives': metrics.TrueNegatives(),
    'FalsePositives': metrics.FalsePositives(),
    'FalseNegatives': metrics.FalseNegatives()
}


callbacks_dict = {
    'ModelCheckpoint': callbacks.ModelCheckpoint(filepath='checkpoints/model.{epoch:02d}-{val_loss:.2f}.h5',
                                                 monitor='val_loss', verbose=1),
    'TensorBoard': callbacks.TensorBoard(log_dir='logs', update_freq='epoch', embeddings_freq=0),
    'EarlyStopping': callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1,
                                             restore_best_weights=False),
    'ReduceLROnPlateau': callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,
                                                     verbose=1),
    'TerminateOnNaN': callbacks.TerminateOnNaN()
}


def create_metrics(metrics_names):
    """
        :param metrics_names: list of metrics names

        :return: list with keras metrics
    """
    array_metrics = []

    for metric in metrics_names:
        array_metrics.append(metrics_dict[metric])

    return array_metrics


def create_callbacks(callbacks_names=None):
    """
        :param callbacks_names:

        :return:
    """
    if not os.path.exists('logs'):
        os.makedirs('logs')

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    if not callbacks_names:
        callbacks_names = ['ModelCheckpoint', 'TensorBoard', 'EarlyStopping']

    array_callbacks = []

    for callback in callbacks_names:
        array_callbacks.append(callbacks_dict[callback])

    return array_callbacks
