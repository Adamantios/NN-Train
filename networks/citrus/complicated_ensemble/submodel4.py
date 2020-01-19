from typing import Union

from numpy.core.multiarray import ndarray
from tensorflow.python.keras import Model

from networks.citrus.complicated_ensemble.submodel1 import citrus_complicated_ensemble_submodel1


def citrus_complicated_ensemble_submodel4(input_shape=None, input_tensor=None, n_classes=None,
                                          weights_path: Union[None, str] = None) -> Model:
    """
    Defines a citrus network.

    :param n_classes: the number of classes.
    :param input_shape: the input shape of the network. Can be omitted if input_tensor is used.
    :param input_tensor: the input tensor of the network. Can be omitted if input_shape is used.
    :param weights_path: a path to a trained custom network's weights.
    :return: Keras functional API Model.
    """
    model = citrus_complicated_ensemble_submodel1(input_shape, input_tensor, n_classes, weights_path)
    model._name = 'citrus_complicated_ensemble_submodel4'
    return model


def citrus_complicated_ensemble_submodel4_labels_manipulation(labels_array: ndarray) -> int:
    """
    The model's labels manipulator.

    :param labels_array: the labels to manipulate.
    :return: the number of classes predicted by the model.
    """
    # Labels 0, 3.
    labels_array[labels_array == 1] = 2
    labels_array[labels_array == 3] = 1
    return 3
