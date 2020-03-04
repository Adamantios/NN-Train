from typing import Union

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Average, Concatenate, \
    Softmax

from networks.cifar10.complicated_ensemble_v2.submodel1 import cifar10_complicated_ensemble_v2_submodel1
from networks.cifar10.complicated_ensemble_v2.submodel2 import cifar10_complicated_ensemble_v2_submodel2
from networks.cifar10.complicated_ensemble_v2.submodel3 import cifar10_complicated_ensemble_v2_submodel3
from networks.cifar10.complicated_ensemble_v2.submodel4 import cifar10_complicated_ensemble_v2_submodel4
from networks.cifar10.complicated_ensemble_v2.submodel5 import cifar10_complicated_ensemble_v2_submodel5
from networks.tools import Crop, load_weights, create_inputs


def cifar10_complicated_ensemble_v2(input_shape=None, input_tensor=None, n_classes=None,
                                    weights_path: Union[None, str] = None) -> Model:
    """
    Defines a cifar10 network.

    :param n_classes: used in order to be compatible with the main script.
    :param input_shape: the input shape of the network. Can be omitted if input_tensor is used.
    :param input_tensor: the input tensor of the network. Can be omitted if input_shape is used.
    :param weights_path: a path to a trained custom network's weights.
    :return: Keras functional API Model.
    """
    outputs_list = []
    inputs = create_inputs(input_shape, input_tensor)

    # Generate Submodels.
    submodel1 = cifar10_complicated_ensemble_v2_submodel1(input_shape, input_tensor, 5, weights_path)
    submodel2 = cifar10_complicated_ensemble_v2_submodel2(input_shape, input_tensor, 5, weights_path)
    submodel3 = cifar10_complicated_ensemble_v2_submodel3(input_shape, input_tensor, 5, weights_path)
    submodel4 = cifar10_complicated_ensemble_v2_submodel4(input_shape, input_tensor, 5, weights_path)
    submodel5 = cifar10_complicated_ensemble_v2_submodel5(input_shape, input_tensor, 5, weights_path)

    # Get their outputs.
    outputs_submodel1 = submodel1(inputs)
    outputs_submodel2 = submodel2(inputs)
    outputs_submodel3 = submodel3(inputs)
    outputs_submodel4 = submodel4(inputs)
    outputs_submodel5 = submodel5(inputs)

    # Correct submodel 2 - 5 outputs.
    outputs_submodel2 = Crop(1, 1, None)(outputs_submodel2)
    outputs_submodel3 = Crop(1, 1, None)(outputs_submodel3)
    outputs_submodel4 = Crop(1, 1, None)(outputs_submodel4)
    outputs_submodel5 = Crop(1, 1, None)(outputs_submodel5)

    # Create the complicated outputs.
    # Class 0.
    outputs_list.append(Average(name='class_0')(
        [
            Crop(1, 0, 1)(outputs_submodel1),
            Crop(1, 1, 2)(outputs_submodel5)
        ]
    ))

    # Classes 1, 2, 3.
    outputs_list.append(Average(name='classes_1_2_3')(
        [
            Crop(1, 1, 4)(outputs_submodel1),
            Crop(1, 0, 3)(outputs_submodel2)
        ]
    ))

    # Class 4.
    outputs_list.append(Average(name='class_4')(
        [
            Crop(1, 3, 4)(outputs_submodel2),
            Crop(1, 0, 1)(outputs_submodel3)
        ]
    ))

    # Class 5.
    outputs_list.append(Average(name='class_5')(
        [
            Crop(1, 1, 2)(outputs_submodel3),
            Crop(1, 0, 1)(outputs_submodel5)
        ]
    ))

    # Classes 6, 7.
    outputs_list.append(Average(name='classes_6_7')(
        [
            Crop(1, 2, 4)(outputs_submodel3),
            Crop(1, 0, 2)(outputs_submodel4)
        ]
    ))

    # Classes 8, 9.
    outputs_list.append(Average(name='classes_8_9')(
        [
            Crop(1, 2, 4)(outputs_submodel4),
            Crop(1, 1, 3)(outputs_submodel5)
        ]
    ))

    # Concatenate all class predictions together.
    outputs = Concatenate(name='output')(outputs_list)
    outputs = Softmax(name='output_softmax')(outputs)

    # Create model.
    model = Model(inputs, outputs, name='cifar10_complicated_ensemble_v2')
    # Load weights, if they exist.
    load_weights(weights_path, model)

    return model
