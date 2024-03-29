from typing import Union

from numpy.core.multiarray import ndarray
from numpy.ma import logical_and
from tensorflow.python.keras import Model

from networks.cifar100.complicated_ensemble.submodel5 import cifar100_complicated_ensemble_submodel5


def cifar100_complicated_ensemble_v2_submodel5(input_shape=None, input_tensor=None, n_classes=None,
                                               weights_path: Union[None, str] = None) -> Model:
    """
    Defines a cifar100 network.

    :param n_classes: used in order to be compatible with the main script.
    :param input_shape: the input shape of the network. Can be omitted if input_tensor is used.
    :param input_tensor: the input tensor of the network. Can be omitted if input_shape is used.
    :param weights_path: a path to a trained custom network's weights.
    :return: Keras functional API Model.
    """
    model = cifar100_complicated_ensemble_submodel5(input_shape, input_tensor, n_classes, weights_path)
    model._name = 'cifar100_complicated_ensemble_v2_submodel5'
    return model


def cifar100_complicated_ensemble_v2_submodel5_labels_manipulation(labels_array: ndarray) -> int:
    """
    The model's labels manipulator.

    :param labels_array: the labels to manipulate.
    :return: the number of classes predicted by the model.
    """
    # 50 - 59, 80 - 99, 0 - 9 classes.
    labels_array[logical_and(labels_array > 9, labels_array < 50)] = -1
    labels_array[logical_and(labels_array > 59, labels_array < 80)] = -1

    for i in range(0, 10):
        labels_array[labels_array == i] = i + 1
        labels_array[labels_array == i + 50] = i + 11
        labels_array[labels_array == i + 80] = i + 21
        labels_array[labels_array == i + 90] = i + 31

    labels_array[labels_array == -1] = 0

    return 41
