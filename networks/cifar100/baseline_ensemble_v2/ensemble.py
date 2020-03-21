from typing import Union

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Average

from networks.cifar100.complicated_ensemble_v2.submodel1 import cifar100_complicated_ensemble_v2_submodel1
from networks.cifar100.complicated_ensemble_v2.submodel2 import cifar100_complicated_ensemble_v2_submodel2
from networks.cifar100.complicated_ensemble_v2.submodel3 import cifar100_complicated_ensemble_v2_submodel3
from networks.cifar100.complicated_ensemble_v2.submodel4 import cifar100_complicated_ensemble_v2_submodel4
from networks.cifar100.complicated_ensemble_v2.submodel5 import cifar100_complicated_ensemble_v2_submodel5
from networks.tools import load_weights, create_inputs


def cifar100_baseline_ensemble_v2(input_shape=None, input_tensor=None, n_classes=None,
                                  weights_path: Union[None, str] = None) -> Model:
    """
    Defines a cifar100_baseline_ensemble_v2 network.

    :param n_classes: used in order to be compatible with the main script.
    :param input_shape: the input shape of the network. Can be omitted if input_tensor is used.
    :param input_tensor: the input tensor of the network. Can be omitted if input_shape is used.
    :param weights_path: a path to a trained custom network's weights.
    :return: Keras functional API Model.
    """
    inputs = create_inputs(input_shape, input_tensor)

    # Generate Submodels.
    submodel_1 = cifar100_complicated_ensemble_v2_submodel1(input_shape, input_tensor, n_classes, weights_path)
    submodel_2 = cifar100_complicated_ensemble_v2_submodel2(input_shape, input_tensor, n_classes, weights_path)
    submodel_3 = cifar100_complicated_ensemble_v2_submodel3(input_shape, input_tensor, n_classes, weights_path)
    submodel_4 = cifar100_complicated_ensemble_v2_submodel4(input_shape, input_tensor, n_classes, weights_path)
    submodel_5 = cifar100_complicated_ensemble_v2_submodel5(input_shape, input_tensor, n_classes, weights_path)

    submodel_1._name = 'cifar100_baseline_ensemble_v2_submodel1'
    submodel_2._name = 'cifar100_baseline_ensemble_v2_submodel2'
    submodel_3._name = 'cifar100_baseline_ensemble_v2_submodel3'

    # Get their outputs.
    outputs_submodel1 = submodel_1(inputs)
    outputs_submodel2 = submodel_2(inputs)
    outputs_submodel3 = submodel_3(inputs)
    outputs_submodel4 = submodel_4(inputs)
    outputs_submodel5 = submodel_5(inputs)

    # Average classes.
    outputs = Average(name='averaged_predictions')([outputs_submodel1, outputs_submodel2, outputs_submodel3,
                                                    outputs_submodel4, outputs_submodel5])

    # Create model.
    model = Model(inputs, outputs, name='cifar100_baseline_ensemble_v2')
    # Load weights, if they exist.
    load_weights(weights_path, model)

    return model
