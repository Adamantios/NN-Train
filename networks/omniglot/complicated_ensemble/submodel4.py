from typing import Union

from numpy.core.multiarray import ndarray
from numpy.ma import logical_or
from tensorflow.python.keras import Model

from networks.omniglot.complicated_ensemble.submodel1 import omniglot_complicated_ensemble_submodel1


def omniglot_complicated_ensemble_submodel4(input_shape=None, input_tensor=None, n_classes=None,
                                            weights_path: Union[None, str] = None) -> Model:
    """
    Defines a omniglot network.

    :param n_classes: the number of classes.
    :param input_shape: the input shape of the network. Can be omitted if input_tensor is used.
    :param input_tensor: the input tensor of the network. Can be omitted if input_shape is used.
    :param weights_path: a path to a trained custom network's weights.
    :return: Keras functional API Model.
    """
    model = omniglot_complicated_ensemble_submodel1(input_shape, input_tensor, n_classes, weights_path)
    model._name = 'omniglot_complicated_ensemble_submodel4'
    return model


def omniglot_complicated_ensemble_submodel4_labels_manipulation(labels_array: ndarray) -> int:
    """
    The model's labels manipulator.

    :param labels_array: the labels to manipulate.
    :return: the number of classes predicted by the model.
    """
    # Labels [974 - 1298].
    labels_array[logical_or(labels_array < 974, labels_array > 1298)] = -1

    for i, label_i in enumerate(range(974, 1299)):
        labels_array[labels_array == label_i] = i

    labels_array[labels_array == -1] = 325

    return 326
