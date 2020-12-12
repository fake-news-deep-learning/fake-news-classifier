from keras import metrics


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

def create_metrics(metrics_names):
    """
        :param metrics_names: list of metrics names

        :return: list with keras metrics
    """
    array_metrics = []

    for metric in metrics_names:
        array_metrics.append(metrics_dict[metric])

    return array_metrics
