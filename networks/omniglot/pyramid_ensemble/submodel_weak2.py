from typing import Union

from numpy.core.multiarray import ndarray
from tensorflow.python.keras import Model

from networks.omniglot.pyramid_ensemble.submodel_weak1 import omniglot_pyramid_ensemble_submodel_weak1


def omniglot_pyramid_ensemble_submodel_weak2(input_shape=None, input_tensor=None, n_classes=None,
                                             weights_path: Union[None, str] = None) -> Model:
    """
    Defines a omniglot network.

    :param n_classes: used in order to be compatible with the main script.
    :param input_shape: the input shape of the network. Can be omitted if input_tensor is used.
    :param input_tensor: the input tensor of the network. Can be omitted if input_shape is used.
    :param weights_path: a path to a trained custom network's weights.
    :return: Keras functional API Model.
    """
    model = omniglot_pyramid_ensemble_submodel_weak1(input_shape, input_tensor, n_classes, weights_path)
    model._name = 'omniglot_pyramid_ensemble_submodel_weak2'

    return model


def omniglot_pyramid_ensemble_submodel_weak2_labels_manipulation(labels_array: ndarray) -> int:
    """
    The model's labels manipulator.

    :param labels_array: the labels to manipulate.
    :return: the number of classes predicted by the model.
    """
    labels_array[labels_array < 812] = -1

    for i, label_i in enumerate(range(812, 1623)):
        labels_array[labels_array == label_i] = i

    labels_array[labels_array == -1] = 811

    return 812
