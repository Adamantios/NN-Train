from typing import Union

from numpy.core.multiarray import ndarray
from numpy.ma import logical_or
from tensorflow.python.keras import Model

from networks.caltech101.complicated_ensemble.submodel1 import caltech_complicated_ensemble_submodel1


def caltech_complicated_ensemble_submodel4(input_shape=None, input_tensor=None, n_classes=None,
                                           weights_path: Union[None, str] = None) -> Model:
    """
    Defines a caltech network.

    :param n_classes: used in order to be compatible with the main script.
    :param input_shape: the input shape of the network. Can be omitted if input_tensor is used.
    :param input_tensor: the input tensor of the network. Can be omitted if input_shape is used.
    :param weights_path: a path to a trained custom network's weights.
    :return: Keras functional API Model.
    """
    model = caltech_complicated_ensemble_submodel1(input_shape, input_tensor, n_classes, weights_path)
    model._name = 'caltech_complicated_ensemble_submodel4'
    return model


def caltech_complicated_ensemble_submodel4_labels_manipulation(labels_array: ndarray) -> int:
    """
    The model's labels manipulator.

    :param labels_array: the labels to manipulate.
    :return: the number of classes predicted by the model.
    """
    labels_array[logical_or(labels_array < 62, labels_array > 81)] = -1

    for i, label_i in enumerate(range(62, 82)):
        labels_array[labels_array == label_i] = i

    labels_array[labels_array == -1] = 20

    return 21
