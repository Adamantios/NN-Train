from typing import Union

from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, InputLayer, Dense, Flatten, Average, Concatenate

from networks.tools import Crop, load_weights


def cifar100_complicated_ensemble(input_shape=None, input_tensor=None, n_classes=None,
                                  weights_path: Union[None, str] = None) -> Model:
    """
    Defines a cifar10 network.

    :param n_classes: used in order to be compatible with the main script.
    :param input_shape: the input shape of the network. Can be omitted if input_tensor is used.
    :param input_tensor: the input tensor of the network. Can be omitted if input_shape is used.
    :param weights_path: a path to a trained custom network's weights.
    :return: Keras functional API Model.
    """
    output_list = []

    if input_shape is None and input_tensor is None:
        raise ValueError('You need to specify input shape or input tensor for the network.')

    # Create input.
    if input_shape is None:
        # Create an InputLayer using the input tensor.
        inputs = InputLayer(input_tensor=input_tensor, name='input')
    else:
        inputs = Input(shape=input_shape, name='input')

    # Submodel 1.
    # Block1.
    x1 = Conv2D(32, (3, 3), padding='same', activation='elu', name='submodel1_block1_conv1')(inputs)
    x1 = Conv2D(32, (3, 3), padding='same', activation='elu', name='submodel1_block1_conv2')(x1)
    x1 = MaxPooling2D(pool_size=(2, 2), name='submodel1_block1_pool')(x1)

    # Block2
    x1 = Conv2D(64, (3, 3), padding='same', activation='elu', name='submodel1_block2_conv1')(x1)
    x1 = Conv2D(64, (3, 3), padding='same', activation='elu', name='submodel1_block2_conv2')(x1)
    x1 = MaxPooling2D(pool_size=(2, 2), name='submodel1_block2_pool')(x1)

    # Block3
    x1 = Conv2D(128, (3, 3), padding='same', activation='elu', name='submodel1_block3_conv1')(x1)
    x1 = Conv2D(128, (3, 3), padding='same', activation='elu', name='submodel1_block3_conv2')(x1)
    x1 = MaxPooling2D(pool_size=(2, 2), name='submodel1_block3_pool')(x1)

    # Add Submodel 1 top layers.
    x1 = Flatten(name='submodel1_flatten')(x1)
    outputs1 = Dense(20, activation='softmax', name='submodel1_softmax')(x1)
    # Crop outputs1 in order to create the first submodel's output.
    outputs_first_submodel = Crop(1, 0, 10, name='first_ten_classes_submodel')(outputs1)
    output_list.append(outputs_first_submodel)

    # Submodel 2.
    # Block1.
    x2 = Conv2D(64, (3, 3), padding='same', activation='elu', name='submodel2_block1_conv1')(inputs)
    x2 = Conv2D(64, (3, 3), padding='same', activation='elu', name='submodel2_block1_conv2')(x2)
    x2 = MaxPooling2D(pool_size=(2, 2), name='submodel2_block1_pool')(x2)

    # Block2
    x2 = Conv2D(128, (3, 3), padding='same', activation='elu', name='submodel2_block2_conv1')(x2)
    x2 = Conv2D(128, (3, 3), padding='same', activation='elu', name='submodel2_block2_conv2')(x2)
    x2 = MaxPooling2D(pool_size=(2, 2), name='submodel2_block2_pool')(x2)

    # Add Submodel 2 top layers.
    x2 = Flatten(name='submodel2_flatten')(x2)
    outputs2 = Dense(30, activation='softmax', name='submodel2_softmax')(x2)

    # Average the predictions for the second class of the first two submodels.
    averaged_classes_20_30 = Average(name='averaged_second_ten_classes')(
        [Crop(1, 10, 20)(outputs1), Crop(1, 0, 10)(outputs2)])
    # Crop outputs2 in order to create the third ten classes output.
    outputs_classes_30_40 = Crop(1, 10, 20, name='third_ten_classes')(outputs2)
    # Concatenate classes outputs in order to create the second submodel's output.
    outputs_second_submodel = Concatenate(name='second_submodel')([averaged_classes_20_30, outputs_classes_30_40])
    output_list.append(outputs_second_submodel)

    # Submodel 3.
    # Block1.
    x3 = Conv2D(64, (3, 3), padding='same', activation='elu', name='submodel3_block1_conv1')(inputs)
    x3 = Conv2D(64, (3, 3), padding='same', activation='elu', name='submodel3_block1_conv2')(x3)
    x3 = MaxPooling2D(pool_size=(2, 2), name='submodel3_block1_pool')(x3)

    # Block2
    x3 = Conv2D(128, (3, 3), padding='same', activation='elu', name='submodel3_block2_conv1')(x3)
    x3 = Conv2D(128, (3, 3), padding='same', activation='elu', name='submodel3_block2_conv2')(x3)
    x3 = MaxPooling2D(pool_size=(2, 2), name='submodel3_block2_pool')(x3)

    # Add Submodel 3 top layers.
    x3 = Flatten(name='submodel3_flatten')(x3)
    outputs3 = Dense(30, activation='softmax', name='submodel3_softmax')(x3)

    # Average the predictions for the fourth class of the last two submodels.
    averaged_classes_30_40 = Average(name='averaged_fourth_ten_class')([
        Crop(1, 20, 30)(outputs2),
        Crop(1, 0, 10)(outputs3)
    ])
    # Crop outputs3 in order to create the fifth abd sixth class outputs.
    outputs_classes_50_60 = Crop(1, 10, 20, name='fifth_ten_class')(outputs3)
    outputs_classes_60_70 = Crop(1, 20, 30, name='sixth_ten_class')(outputs3)
    # Concatenate classes outputs in order to create the third submodel's output.
    outputs_third_submodel = Concatenate(name='third_submodel')([
        averaged_classes_30_40,
        outputs_classes_50_60,
        outputs_classes_60_70
    ])
    output_list.append(outputs_third_submodel)

    # Submodel 4.
    # Block1.
    x4 = Conv2D(64, (3, 3), padding='same', activation='elu', name='submodel4_block1_conv1')(inputs)
    x4 = Conv2D(64, (3, 3), padding='same', activation='elu', name='submodel4_block1_conv2')(x4)
    x4 = MaxPooling2D(pool_size=(2, 2), name='submodel4_block1_pool')(x4)

    # Block2
    x4 = Conv2D(64, (3, 3), padding='same', activation='elu', name='submodel4_block2_conv1')(x4)
    x4 = Conv2D(64, (3, 3), padding='same', activation='elu', name='submodel4_block2_conv2')(x4)
    x4 = MaxPooling2D(pool_size=(2, 2), name='submodel4_block2_pool')(x4)

    # Block3
    x4 = Conv2D(128, (3, 3), padding='same', activation='elu', name='submodel4_block3_conv1')(x4)
    x4 = Conv2D(128, (3, 3), padding='same', activation='elu', name='submodel4_block3_conv2')(x4)
    x4 = MaxPooling2D(pool_size=(2, 2), name='submodel4_block3_pool')(x4)

    # Add Submodel 4 top layers.
    x4 = Flatten(name='submodel4_flatten')(x4)
    outputs4 = Dense(20, activation='softmax', name='seventh_eighth_class_submodel4')(x4)
    output_list.append(outputs4)

    # Submodel 5.
    # Block1.
    x5 = Conv2D(32, (3, 3), padding='same', activation='elu', name='submodel5_block1_conv1')(inputs)
    x5 = Conv2D(32, (3, 3), padding='same', activation='elu', name='submodel5_block1_conv2')(x5)
    x5 = MaxPooling2D(pool_size=(2, 2), name='submodel5_block1_pool')(x5)

    # Block2
    x5 = Conv2D(64, (3, 3), padding='same', activation='elu', name='submodel5_block2_conv1')(x5)
    x5 = Conv2D(64, (3, 3), padding='same', activation='elu', name='submodel5_block2_conv2')(x5)
    x5 = MaxPooling2D(pool_size=(2, 2), name='submodel5_block2_pool')(x5)

    # Block3
    x5 = Conv2D(128, (3, 3), padding='same', activation='elu', name='submodel5_block3_conv1')(x5)
    x5 = Conv2D(128, (3, 3), padding='same', activation='elu', name='submodel5_block3_conv2')(x5)
    x5 = MaxPooling2D(pool_size=(2, 2), name='submodel5_block3_pool')(x5)

    # Add Submodel 5 top layers.
    x5 = Flatten(name='submodel5_flatten')(x5)
    outputs5 = Dense(20, activation='softmax', name='ninth_tenth_class_submodel5')(x5)
    output_list.append(outputs5)

    # Concatenate all class predictions together.
    outputs = Concatenate(name='output')(output_list)

    # Create model.
    model = Model(inputs, outputs, name='cifar100_complicated_ensemble')
    # Load weights, if they exist.
    load_weights(weights_path, model)

    return model
