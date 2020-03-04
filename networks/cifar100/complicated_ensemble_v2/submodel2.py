from typing import Union

from numpy.core.multiarray import ndarray
from numpy.ma import logical_or
from tensorflow.python.keras import Model

from networks.cifar100.complicated_ensemble.submodel2 import cifar100_complicated_ensemble_submodel2


def cifar100_complicated_ensemble_v2_submodel2(input_shape=None, input_tensor=None, n_classes=None,
                                               weights_path: Union[None, str] = None) -> Model:
    """
    Defines a cifar100 network.

    :param n_classes: used in order to be compatible with the main script.
    :param input_shape: the input shape of the network. Can be omitted if input_tensor is used.
    :param input_tensor: the input tensor of the network. Can be omitted if input_shape is used.
    :param weights_path: a path to a trained custom network's weights.
    :return: Keras functional API Model.
    """
    model = cifar100_complicated_ensemble_submodel2(input_shape, input_tensor, n_classes, weights_path)
    model._name = 'cifar100_complicated_ensemble_v2_submodel2'
    return model


def cifar100_complicated_ensemble_v2_submodel2_labels_manipulation(labels_array: ndarray) -> int:
    """
    The model's labels manipulator.

    :param labels_array: the labels to manipulate.
    :return: the number of classes predicted by the model.
    """
    # 10 - 49 classes.
    labels_array[logical_or(labels_array < 10, labels_array > 49)] = 0
    return 41
