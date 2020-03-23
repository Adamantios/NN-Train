from typing import Union

from tensorflow.python.keras import Model

from networks.cifar100.complicated_ensemble_v2.submodel2 import cifar100_complicated_ensemble_v2_submodel2


def cifar100_baseline_ensemble_v2_submodel2(input_shape=None, input_tensor=None, n_classes=None,
                                            weights_path: Union[None, str] = None) -> Model:
    """
    Defines a cifar100 network.

    :param n_classes: used in order to be compatible with the main script.
    :param input_shape: the input shape of the network. Can be omitted if input_tensor is used.
    :param input_tensor: the input tensor of the network. Can be omitted if input_shape is used.
    :param weights_path: a path to a trained custom network's weights.
    :return: Keras functional API Model.
    """
    model = cifar100_complicated_ensemble_v2_submodel2(input_shape, input_tensor, n_classes, weights_path)
    model._name = 'cifar100_baseline_ensemble_v2_submodel2'
    return model
