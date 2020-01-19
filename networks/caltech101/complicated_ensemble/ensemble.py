from typing import Union

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, Average, Concatenate, \
    Softmax

from networks.caltech101.complicated_ensemble.submodel1 import caltech_complicated_ensemble_submodel1
from networks.caltech101.complicated_ensemble.submodel2 import caltech_complicated_ensemble_submodel2
from networks.caltech101.complicated_ensemble.submodel3 import caltech_complicated_ensemble_submodel3
from networks.caltech101.complicated_ensemble.submodel4 import caltech_complicated_ensemble_submodel4
from networks.caltech101.complicated_ensemble.submodel5 import caltech_complicated_ensemble_submodel5
from networks.tools import Crop, load_weights, create_inputs


def caltech_complicated_ensemble(input_shape=None, input_tensor=None, n_classes=None,
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

    # Submodel 1.
    submodel1 = caltech_complicated_ensemble_submodel1(input_shape, input_tensor, n_classes, weights_path)
    outputs1 = Dense(22, name='submodel1_output')(submodel1.layers[-2])
    # Crop outputs1 in order to create the first submodel's output.
    outputs_first_submodel = Crop(1, 0, 12, name='first_twelve_classes_submodel')(outputs1)
    output_list.append(outputs_first_submodel)

    # Submodel 2.
    submodel2 = caltech_complicated_ensemble_submodel2(input_shape, input_tensor, n_classes, weights_path)
    outputs2 = Dense(30, name='submodel2_output')(submodel2.layers[-2])

    # Average the predictions for the second class of the first two submodels.
    averaged_classes_20_30 = Average(name='averaged_second_ten_classes')(
        [Crop(1, 12, 22)(outputs1), Crop(1, 0, 10)(outputs2)])
    # Crop outputs2 in order to create the third ten classes output.
    outputs_classes_30_40 = Crop(1, 10, 20, name='third_ten_classes')(outputs2)
    # Concatenate classes outputs in order to create the second submodel's output.
    outputs_second_submodel = Concatenate(name='second_submodel')([averaged_classes_20_30, outputs_classes_30_40])
    output_list.append(outputs_second_submodel)

    # Submodel 3.
    submodel3 = caltech_complicated_ensemble_submodel3(input_shape, input_tensor, n_classes, weights_path)
    outputs3 = Dense(30, name='submodel3_output')(submodel3.layers[-2])

    # Average the predictions for the fourth class of the last two submodels.
    averaged_classes_30_40 = Average(name='averaged_fourth_ten_class')([
        Crop(1, 20, 30)(outputs2),
        Crop(1, 0, 10)(outputs3)
    ])
    # Crop outputs3 in order to create the fifth abd sixth class outputs.
    outputs_classes_40_50 = Crop(1, 10, 20, name='fifth_ten_class')(outputs3)
    outputs_classes_50_60 = Crop(1, 20, 30, name='sixth_ten_class')(outputs3)
    # Concatenate classes outputs in order to create the third submodel's output.
    outputs_third_submodel = Concatenate(name='third_submodel')([
        averaged_classes_30_40,
        outputs_classes_40_50,
        outputs_classes_50_60
    ])
    output_list.append(outputs_third_submodel)

    # Submodel 4.
    submodel4 = caltech_complicated_ensemble_submodel4(input_shape, input_tensor, n_classes, weights_path)
    outputs4 = Dense(20, name='submodel4_output')(submodel4.layers[-2])
    output_list.append(outputs4)

    # Submodel 5.
    submodel5 = caltech_complicated_ensemble_submodel5(input_shape, input_tensor, n_classes, weights_path)
    outputs5 = Dense(20, name='submodel5_output')(submodel5.layers[-2])
    output_list.append(outputs5)

    # Concatenate all class predictions together.
    outputs = Concatenate(name='output')(output_list)
    outputs = Softmax(name='output_softmax')(outputs)

    # Create model.
    model = Model(inputs, outputs, name='caltech_complicated_ensemble')
    # Load weights, if they exist.
    load_weights(weights_path, model)

    return model
