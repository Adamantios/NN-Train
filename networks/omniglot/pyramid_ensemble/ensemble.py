from typing import Union

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Average, Concatenate, \
    Softmax

from networks.omniglot.pyramid_ensemble.submodel_strong import omniglot_pyramid_ensemble_submodel_strong
from networks.omniglot.pyramid_ensemble.submodel_weak1 import omniglot_pyramid_ensemble_submodel_weak1
from networks.omniglot.pyramid_ensemble.submodel_weak2 import omniglot_pyramid_ensemble_submodel_weak2
from networks.tools import Crop, load_weights, create_inputs


# TODO follow the same code style, in order to improve the other datasets complicated ensembles too.
def omniglot_pyramid_ensemble(input_shape=None, input_tensor=None, n_classes=None,
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
    submodel_strong = omniglot_pyramid_ensemble_submodel_strong(input_shape, input_tensor, n_classes, weights_path)
    submodel_weak1 = omniglot_pyramid_ensemble_submodel_weak1(input_shape, input_tensor, 3, weights_path)
    submodel_weak2 = omniglot_pyramid_ensemble_submodel_weak2(input_shape, input_tensor, 3, weights_path)
    # Get their outputs.
    outputs_submodel_strong = submodel_strong.output
    outputs_submodel_weak1 = submodel_weak1.output
    outputs_submodel_weak2 = submodel_weak2.output

    # Average classes.
    first_classes = Average(name='averaged_first_classes')(
        [
            Crop(1, 0, 812, name='first_classes_submodel_strong')(outputs_submodel_strong),
            Crop(1, 0, 812, name='first_classes_submodel_weak1')(outputs_submodel_weak1)
        ]
    )

    last_classes = Average(name='averaged_last_classes')(
        [
            Crop(1, 812, 1623, name='last_classes_submodel_strong')(outputs_submodel_strong),
            Crop(1, 0, 811, name='last_classes_submodel_weak2')(outputs_submodel_weak2)
        ]
    )

    # Concatenate all class predictions together.
    outputs = Concatenate(name='output')([first_classes, last_classes])
    outputs = Softmax(name='output_softmax')(outputs)

    # Create model.
    model = Model(inputs, outputs, name='omniglot_pyramid_ensemble')
    # Load weights, if they exist.
    load_weights(weights_path, model)

    return model
