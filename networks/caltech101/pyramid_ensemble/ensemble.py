from typing import Union

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, Average, Concatenate, \
    Softmax

from networks.caltech101.pyramid_ensemble.submodel_strong import caltech_pyramid_ensemble_submodel_strong
from networks.caltech101.pyramid_ensemble.submodel_weak1 import caltech_pyramid_ensemble_submodel_weak1
from networks.caltech101.pyramid_ensemble.submodel_weak2 import caltech_pyramid_ensemble_submodel_weak2
from networks.tools import Crop, load_weights, create_inputs


def caltech_pyramid_ensemble(input_shape=None, input_tensor=None, n_classes=None,
                             weights_path: Union[None, str] = None) -> Model:
    """
    Defines a caltech network.

    :param n_classes: used in order to be compatible with the main script.
    :param input_shape: the input shape of the network. Can be omitted if input_tensor is used.
    :param input_tensor: the input tensor of the network. Can be omitted if input_shape is used.
    :param weights_path: a path to a trained custom network's weights.
    :return: Keras functional API Model.
    """
    output_list = []
    inputs = create_inputs(input_shape, input_tensor)

    # Submodel Strong.
    submodel_strong = caltech_pyramid_ensemble_submodel_strong(input_shape, input_tensor, n_classes, weights_path)
    outputs_submodel_strong = submodel_strong.output

    # Submodel Weak 1.
    submodel_weak1 = caltech_pyramid_ensemble_submodel_weak1(input_shape, input_tensor, n_classes, weights_path)
    outputs_submodel_weak1 = Dense(61, name='outputs_submodel_weak1')(submodel_weak1.layers[-1])

    # Average the predictions for the first half classes.
    averaged_first_half_classes = Average(name='averaged_first_half_classes')(
        [
            Crop(1, 0, 61)(outputs_submodel_strong),
            outputs_submodel_weak1
        ]
    )

    output_list.append(averaged_first_half_classes)

    # Submodel Weak 2.
    # Block1.
    submodel_weak2 = caltech_pyramid_ensemble_submodel_weak2(input_shape, input_tensor, n_classes, weights_path)
    outputs_submodel_weak2 = Dense(61, name='outputs_submodel_weak2')(submodel_weak2.layers[-1])

    # Average the predictions for the last half classes.
    averaged_last_half_classes = Average(name='averaged_last_half_classes')(
        [
            Crop(1, 61, 102)(outputs_submodel_strong),
            outputs_submodel_weak2
        ]
    )

    output_list.append(averaged_last_half_classes)

    # Concatenate all class predictions together.
    outputs = Concatenate(name='output')(output_list)
    outputs = Softmax(name='output_softmax')(outputs)

    # Create model.
    model = Model(inputs, outputs, name='caltech_pyramid_ensemble')
    # Load weights, if they exist.
    load_weights(weights_path, model)

    return model
