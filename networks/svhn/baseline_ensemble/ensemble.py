from typing import Union

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Average

from networks.svhn.different_architectures.model1 import svhn_model1
from networks.tools import load_weights, create_inputs


def svhn_baseline_ensemble(input_shape=None, input_tensor=None, n_classes=None,
                           weights_path: Union[None, str] = None) -> Model:
    """
    Defines a svhn_baseline_ensemble network.

    :param n_classes: used in order to be compatible with the main script.
    :param input_shape: the input shape of the network. Can be omitted if input_tensor is used.
    :param input_tensor: the input tensor of the network. Can be omitted if input_shape is used.
    :param weights_path: a path to a trained custom network's weights.
    :return: Keras functional API Model.
    """
    inputs = create_inputs(input_shape, input_tensor)

    # Generate Submodels.
    submodel_1 = svhn_model1(n_classes, input_shape, input_tensor, weights_path)
    submodel_2 = svhn_model1(n_classes, input_shape, input_tensor, weights_path)
    submodel_3 = svhn_model1(n_classes, input_shape, input_tensor, weights_path)

    submodel_1._name = 'svhn_baseline_ensemble_submodel1'
    submodel_2._name = 'svhn_baseline_ensemble_submodel2'
    submodel_3._name = 'svhn_baseline_ensemble_submodel3'

    # Get their outputs.
    outputs_submodel1 = submodel_1(inputs)
    outputs_submodel2 = submodel_2(inputs)
    outputs_submodel3 = submodel_3(inputs)

    # Average classes.
    outputs = Average(name='averaged_predictions')([outputs_submodel1, outputs_submodel2, outputs_submodel3])

    # Create model.
    model = Model(inputs, outputs, name='svhn_baseline_ensemble')
    # Load weights, if they exist.
    load_weights(weights_path, model)

    return model
