from typing import Union

from numpy.core.multiarray import ndarray
from numpy.ma import logical_and
from tensorflow.python.keras import Model

from networks.cifar10.complicated_ensemble.submodel5 import cifar10_complicated_ensemble_submodel5


def cifar10_complicated_ensemble_v2_submodel5(input_shape=None, input_tensor=None, n_classes=None,
                                              weights_path: Union[None, str] = None) -> Model:
    """
    Defines a cifar10 network.

    :param n_classes: used in order to be compatible with the main script.
    :param input_shape: the input shape of the network. Can be omitted if input_tensor is used.
    :param input_tensor: the input tensor of the network. Can be omitted if input_shape is used.
    :param weights_path: a path to a trained custom network's weights.
    :return: Keras functional API Model.
    """
    model = cifar10_complicated_ensemble_submodel5(input_shape, input_tensor, n_classes, weights_path)
    model._name = 'cifar10_complicated_ensemble_v2_submodel5'
    return model


def cifar10_complicated_ensemble_v2_submodel5_labels_manipulation(labels_array: ndarray) -> int:
    """
    The model's labels manipulator.

    :param labels_array: the labels to manipulate.
    :return: the number of classes predicted by the model.
    """
    # 5, 8, 9, 0 classes.
    labels_array[logical_and(labels_array < 5, labels_array > 0)] = 6
    labels_array[labels_array == 0] = 1
    labels_array[labels_array == 5] = 2
    labels_array[labels_array == 8] = 3
    labels_array[labels_array == 9] = 4
    labels_array[labels_array == 6] = 0
    labels_array[labels_array == 7] = 0
    return 5
