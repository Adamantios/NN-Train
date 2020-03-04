from typing import Union

from numpy.core.multiarray import ndarray
from numpy.ma import logical_or
from tensorflow.python.keras import Model

from networks.svhn.complicated_ensemble.submodel3 import svhn_complicated_ensemble_submodel3


def svhn_complicated_ensemble_v2_submodel3(input_shape=None, input_tensor=None, n_classes=None,
                                           weights_path: Union[None, str] = None) -> Model:
    """
    Defines a svhn network.

    :param n_classes: used in order to be compatible with the main script.
    :param input_shape: the input shape of the network. Can be omitted if input_tensor is used.
    :param input_tensor: the input tensor of the network. Can be omitted if input_shape is used.
    :param weights_path: a path to a trained custom network's weights.
    :return: Keras functional API Model.
    """
    return svhn_complicated_ensemble_submodel3(input_shape, input_tensor, n_classes, weights_path)


def svhn_complicated_ensemble_v2_submodel3_labels_manipulation(labels_array: ndarray) -> int:
    """
    The model's labels manipulator.

    :param labels_array: the labels to manipulate.
    :return: the number of classes predicted by the model.
    """
    # 4, 5, 6, 7 classes.
    labels_array[logical_or(labels_array < 4, labels_array > 7)] = 0
    labels_array[labels_array == 4] = 1
    labels_array[labels_array == 5] = 2
    labels_array[labels_array == 6] = 3
    labels_array[labels_array == 7] = 4
    return 5
