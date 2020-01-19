from typing import Union

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Average, Concatenate, \
    Softmax

from networks.citrus.complicated_ensemble.submodel1 import citrus_complicated_ensemble_submodel1
from networks.citrus.complicated_ensemble.submodel2 import citrus_complicated_ensemble_submodel2
from networks.citrus.complicated_ensemble.submodel3 import citrus_complicated_ensemble_submodel3
from networks.citrus.complicated_ensemble.submodel4 import citrus_complicated_ensemble_submodel4
from networks.citrus.complicated_ensemble.submodel5 import citrus_complicated_ensemble_submodel5
from networks.tools import Crop, load_weights, create_inputs


# TODO follow the same code style, in order to improve the other datasets complicated ensembles too.
def citrus_complicated_ensemble(input_shape=None, input_tensor=None, n_classes=None,
                                weights_path: Union[None, str] = None) -> Model:
    """
    Defines a citrus network.

    :param n_classes: used in order to be compatible with the main script.
    :param input_shape: the input shape of the network. Can be omitted if input_tensor is used.
    :param input_tensor: the input tensor of the network. Can be omitted if input_shape is used.
    :param weights_path: a path to a trained custom network's weights.
    :return: Keras functional API Model.
    """
    inputs = create_inputs(input_shape, input_tensor)

    # Generate Submodels.
    submodel1 = citrus_complicated_ensemble_submodel1(input_shape, input_tensor, 3, weights_path)
    submodel2 = citrus_complicated_ensemble_submodel2(input_shape, input_tensor, 3, weights_path)
    submodel3 = citrus_complicated_ensemble_submodel3(input_shape, input_tensor, 4, weights_path)
    submodel4 = citrus_complicated_ensemble_submodel4(input_shape, input_tensor, 3, weights_path)
    submodel5 = citrus_complicated_ensemble_submodel5(input_shape, input_tensor, 4, weights_path)
    # Get their outputs.
    outputs1 = submodel1.output
    outputs2 = submodel2.output
    outputs3 = submodel3.output
    outputs4 = submodel4.output
    outputs5 = submodel5.output

    # Average classes.
    first_class = Average(name='averaged_first_class')(
        [
            Crop(1, 0, 1, name='first_class_submodel1')(outputs1),
            Crop(1, 0, 1, name='first_class_submodel3')(outputs3),
            Crop(1, 0, 1, name='first_class_submodel4')(outputs4)
        ]
    )

    second_class = Average(name='averaged_second_class')(
        [
            Crop(1, 1, 2, name='second_class_submodel1')(outputs1),
            Crop(1, 0, 1, name='second_class_submodel2')(outputs2),
            Crop(1, 1, 2, name='second_class_submodel3')(outputs3),
            Crop(1, 0, 1, name='second_class_submodel5')(outputs5)
        ]
    )

    third_class = Average(name='averaged_third_class')(
        [
            Crop(1, 1, 2, name='third_class_submodel2')(outputs2),
            Crop(1, 2, 3, name='third_class_submodel3')(outputs3),
            Crop(1, 1, 2, name='third_class_submodel5')(outputs5)
        ]
    )

    fourth_class = Average(name='averaged_fourth_class')(
        [
            Crop(1, 1, 2, name='fourth_class_submodel4')(outputs4),
            Crop(1, 2, 3, name='fourth_class_submodel5')(outputs5)
        ]
    )

    # Concatenate all class predictions together.
    outputs = Concatenate(name='output')([first_class, second_class, third_class, fourth_class])
    outputs = Softmax(name='output_softmax')(outputs)

    # Create model.
    model = Model(inputs, outputs, name='citrus_complicated_ensemble')
    # Load weights, if they exist.
    load_weights(weights_path, model)

    return model
