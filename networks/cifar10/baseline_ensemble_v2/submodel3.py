from typing import Union

from tensorflow.python.keras import Model

from networks.cifar10.complicated_ensemble_v2.submodel3 import cifar10_complicated_ensemble_v2_submodel3


def cifar10_baseline_ensemble_v2_submodel3(input_shape=None, input_tensor=None, n_classes=None,
                                           weights_path: Union[None, str] = None) -> Model:
    """
    Defines a cifar10 network.

    :param n_classes: used in order to be compatible with the main script.
    :param input_shape: the input shape of the network. Can be omitted if input_tensor is used.
    :param input_tensor: the input tensor of the network. Can be omitted if input_shape is used.
    :param weights_path: a path to a trained custom network's weights.
    :return: Keras functional API Model.
    """
    model = cifar10_complicated_ensemble_v2_submodel3(input_shape, input_tensor, n_classes, weights_path)
    model._name = 'cifar10_baseline_ensemble_v2_submodel3'
    return model
