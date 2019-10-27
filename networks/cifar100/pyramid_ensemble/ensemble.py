from typing import Union

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Average, Concatenate, \
    Softmax, Dropout, BatchNormalization

from networks.tools import Crop, load_weights, create_inputs


def cifar100_pyramid_ensemble(input_shape=None, input_tensor=None, n_classes=None,
                              weights_path: Union[None, str] = None) -> Model:
    """
    Defines a cifar100 network.

    :param n_classes: used in order to be compatible with the main script.
    :param input_shape: the input shape of the network. Can be omitted if input_tensor is used.
    :param input_tensor: the input tensor of the network. Can be omitted if input_shape is used.
    :param weights_path: a path to a trained custom network's weights.
    :return: Keras functional API Model.
    """
    output_list = []
    inputs = create_inputs(input_shape, input_tensor)

    # Submodel Strong.
    # Block1.
    x1 = Conv2D(64, (3, 3), padding='same', activation='elu', name='submodel_strong_block1_conv1')(inputs)
    x1 = Conv2D(64, (3, 3), padding='same', activation='elu', name='submodel_strong_block1_conv2')(x1)
    x1 = MaxPooling2D(pool_size=(2, 2), name='submodel_strong_block1_pool')(x1)

    # Block2
    x1 = Conv2D(128, (3, 3), padding='same', activation='elu', name='submodel_strong_block2_conv1')(x1)
    x1 = Conv2D(128, (3, 3), padding='same', activation='elu', name='submodel_strong_block2_conv2')(x1)
    x1 = MaxPooling2D(pool_size=(2, 2), name='submodel_strong_block2_pool')(x1)

    # Block3
    x1 = BatchNormalization(name='submodel_strong_block3_batch-norm')(x1)
    x1 = Conv2D(256, (3, 3), padding='same', activation='elu', name='submodel_strong_block3_conv')(x1)
    x1 = Dropout(0.5, name='submodel_strong_block3_dropout')(x1)

    # Add Submodel Strong top layers.
    x1 = Flatten(name='submodel_strong_flatten')(x1)
    outputs_submodel_strong = Dense(100, name='submodel_strong_output')(x1)

    # Submodel Weak 1.
    # Block1.
    x2 = Conv2D(128, (3, 3), padding='same', activation='elu', name='submodel_weak_1_block1_conv1')(inputs)
    x2 = Conv2D(128, (3, 3), padding='same', activation='elu', name='submodel_weak_1_block1_conv2')(x2)
    x2 = MaxPooling2D(pool_size=(2, 2), name='submodel_weak_1_block1_pool')(x2)

    # Add Submodel Weak 1 top layers.
    x2 = Flatten(name='submodel_weak_1_flatten')(x2)
    outputs2 = Dense(50, name='submodel_weak_1_output')(x2)

    # Average the predictions for the first five classes.
    averaged_first_half_classes = Average(name='averaged_first_half_classes')(
        [
            Crop(1, 0, 50)(outputs_submodel_strong),
            outputs2
        ]
    )

    output_list.append(averaged_first_half_classes)

    # Submodel Weak 2.
    # Block1.
    x3 = Conv2D(128, (3, 3), padding='same', activation='elu', name='submodel_weak_2_block1_conv1')(inputs)
    x3 = Conv2D(128, (3, 3), padding='same', activation='elu', name='submodel_weak_2_block1_conv2')(x3)
    x3 = MaxPooling2D(pool_size=(2, 2), name='submodel_weak_2_block1_pool')(x3)

    # Add Submodel Weak 2 top layers.
    x3 = Flatten(name='submodel_weak_2_flatten')(x3)
    outputs3 = Dense(50, name='submodel_weak_2_output')(x3)

    # Average the predictions for the last five classes.
    averaged_last_half_classes = Average(name='averaged_last_half_classes')(
        [
            Crop(1, 50, 100)(outputs_submodel_strong),
            outputs3
        ]
    )

    output_list.append(averaged_last_half_classes)

    # Concatenate all class predictions together.
    outputs = Concatenate(name='output')(output_list)
    outputs = Softmax(name='output_softmax')(outputs)

    # Create model.
    model = Model(inputs, outputs, name='cifar100_pyramid_ensemble')
    # Load weights, if they exist.
    load_weights(weights_path, model)

    return model
