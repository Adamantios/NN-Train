from typing import Union

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Concatenate, \
    Softmax

from networks.omniglot.complicated_ensemble.submodel1 import omniglot_complicated_ensemble_submodel1
from networks.omniglot.complicated_ensemble.submodel2 import omniglot_complicated_ensemble_submodel2
from networks.omniglot.complicated_ensemble.submodel3 import omniglot_complicated_ensemble_submodel3
from networks.omniglot.complicated_ensemble.submodel4 import omniglot_complicated_ensemble_submodel4
from networks.omniglot.complicated_ensemble.submodel5 import omniglot_complicated_ensemble_submodel5
from networks.tools import load_weights, create_inputs


def omniglot_complicated_ensemble(input_shape=None, input_tensor=None, n_classes=None,
                                  weights_path: Union[None, str] = None) -> Model:
    """
    Defines a omniglot network.

    :param n_classes: used in order to be compatible with the main script.
    :param input_shape: the input shape of the network. Can be omitted if input_tensor is used.
    :param input_tensor: the input tensor of the network. Can be omitted if input_shape is used.
    :param weights_path: a path to a trained custom network's weights.
    :return: Keras functional API Model.
    """
    inputs = create_inputs(input_shape, input_tensor)

    # Generate Submodels.
    submodel1 = omniglot_complicated_ensemble_submodel1(input_shape, input_tensor, 325, weights_path)
    submodel2 = omniglot_complicated_ensemble_submodel2(input_shape, input_tensor, 326, weights_path)
    submodel3 = omniglot_complicated_ensemble_submodel3(input_shape, input_tensor, 326, weights_path)
    submodel4 = omniglot_complicated_ensemble_submodel4(input_shape, input_tensor, 326, weights_path)
    submodel5 = omniglot_complicated_ensemble_submodel5(input_shape, input_tensor, 325, weights_path)
    # Get their outputs.
    outputs1 = submodel1.output
    outputs2 = submodel2.output
    outputs3 = submodel3.output
    outputs4 = submodel4.output
    outputs5 = submodel5.output

    # Concatenate all class predictions together.
    outputs = Concatenate(name='output')([outputs1, outputs2, outputs3, outputs4, outputs5])
    outputs = Softmax(name='output_softmax')(outputs)

    # Create model.
    model = Model(inputs, outputs, name='omniglot_complicated_ensemble')
    # Load weights, if they exist.
    load_weights(weights_path, model)

    return model
