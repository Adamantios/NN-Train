from typing import Union

from numpy.core.multiarray import ndarray
from tensorflow.python.keras import Model

from networks.citrus.pyramid_ensemble.submodel_weak1 import citrus_pyramid_ensemble_submodel_weak1


def citrus_pyramid_ensemble_submodel_weak2(input_shape=None, input_tensor=None, n_classes=None,
                                           weights_path: Union[None, str] = None) -> Model:
    """
    Defines a citrus network.

    :param n_classes: used in order to be compatible with the main script.
    :param input_shape: the input shape of the network. Can be omitted if input_tensor is used.
    :param input_tensor: the input tensor of the network. Can be omitted if input_shape is used.
    :param weights_path: a path to a trained custom network's weights.
    :return: Keras functional API Model.
    """
    model = citrus_pyramid_ensemble_submodel_weak1(input_shape, input_tensor, n_classes, weights_path)
    model._name = 'citrus_pyramid_ensemble_submodel_weak2'

    return model


def citrus_pyramid_ensemble_submodel_weak2_labels_manipulation(labels_array: ndarray) -> int:
    """
    The model's labels manipulator.

    :param labels_array: the labels to manipulate.
    :return: the number of classes predicted by the model.
    """
    labels_array[labels_array < 2] = 0
    labels_array[labels_array == 2] = 1
    labels_array[labels_array == 3] = 2
    return 3
