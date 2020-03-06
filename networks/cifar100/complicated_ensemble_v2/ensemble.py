from typing import Union

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Average, Concatenate, Softmax

from networks.cifar100.complicated_ensemble_v2.submodel1 import cifar100_complicated_ensemble_v2_submodel1
from networks.cifar100.complicated_ensemble_v2.submodel2 import cifar100_complicated_ensemble_v2_submodel2
from networks.cifar100.complicated_ensemble_v2.submodel3 import cifar100_complicated_ensemble_v2_submodel3
from networks.cifar100.complicated_ensemble_v2.submodel4 import cifar100_complicated_ensemble_v2_submodel4
from networks.cifar100.complicated_ensemble_v2.submodel5 import cifar100_complicated_ensemble_v2_submodel5
from networks.tools import Crop, load_weights, create_inputs


def cifar100_complicated_ensemble_v2(input_shape=None, input_tensor=None, n_classes=None,
                                     weights_path: Union[None, str] = None) -> Model:
    """
    Defines a cifar100 network.

    :param n_classes: used in order to be compatible with the main script.
    :param input_shape: the input shape of the network. Can be omitted if input_tensor is used.
    :param input_tensor: the input tensor of the network. Can be omitted if input_shape is used.
    :param weights_path: a path to a trained custom network's weights.
    :return: Keras functional API Model.
    """
    outputs_list = []
    inputs = create_inputs(input_shape, input_tensor)

    # Generate Submodels.
    submodel1 = cifar100_complicated_ensemble_v2_submodel1(input_shape, input_tensor, 41, weights_path)
    submodel2 = cifar100_complicated_ensemble_v2_submodel2(input_shape, input_tensor, 41, weights_path)
    submodel3 = cifar100_complicated_ensemble_v2_submodel3(input_shape, input_tensor, 41, weights_path)
    submodel4 = cifar100_complicated_ensemble_v2_submodel4(input_shape, input_tensor, 41, weights_path)
    submodel5 = cifar100_complicated_ensemble_v2_submodel5(input_shape, input_tensor, 41, weights_path)

    # Get their outputs.
    outputs_submodel1 = submodel1(inputs)
    outputs_submodel2 = submodel2(inputs)
    outputs_submodel3 = submodel3(inputs)
    outputs_submodel4 = submodel4(inputs)
    outputs_submodel5 = submodel5(inputs)

    # Correct submodel 2 - 5 outputs.
    outputs_submodel2 = Crop(1, 1, outputs_submodel2.shape[1])(outputs_submodel2)
    outputs_submodel3 = Crop(1, 1, outputs_submodel3.shape[1])(outputs_submodel3)
    outputs_submodel4 = Crop(1, 1, outputs_submodel4.shape[1])(outputs_submodel4)
    outputs_submodel5 = Crop(1, 1, outputs_submodel5.shape[1])(outputs_submodel5)

    # Create the complicated outputs.
    # Classes 0-9.
    outputs_list.append(Average(name='classes_0-9')(
        [
            Crop(1, 0, 10)(outputs_submodel1),
            Crop(1, 10, 20)(outputs_submodel5)
        ]
    ))

    # Classes 10-39.
    outputs_list.append(Average(name='classes_10-39')(
        [
            Crop(1, 10, 40)(outputs_submodel1),
            Crop(1, 0, 30)(outputs_submodel2)
        ]
    ))

    # Classes 40-49.
    outputs_list.append(Average(name='classes_40-49')(
        [
            Crop(1, 30, 40)(outputs_submodel2),
            Crop(1, 0, 10)(outputs_submodel3)
        ]
    ))

    # Classes 50-59.
    outputs_list.append(Average(name='classes_50-59')(
        [
            Crop(1, 10, 20)(outputs_submodel3),
            Crop(1, 0, 10)(outputs_submodel5)
        ]
    ))

    # Classes 60-79.
    outputs_list.append(Average(name='classes_60-79')(
        [
            Crop(1, 20, 40)(outputs_submodel3),
            Crop(1, 0, 20)(outputs_submodel4)
        ]
    ))

    # Classes 80-99.
    outputs_list.append(Average(name='classes_80-99')(
        [
            Crop(1, 20, 40)(outputs_submodel4),
            Crop(1, 10, 30)(outputs_submodel5)
        ]
    ))

    # Concatenate all class predictions together.
    outputs = Concatenate(name='output')(outputs_list)
    outputs = Softmax(name='output_softmax')(outputs)

    # Create model.
    model = Model(inputs, outputs, name='cifar100_complicated_ensemble_v2')
    # Load weights, if they exist.
    load_weights(weights_path, model)

    return model
