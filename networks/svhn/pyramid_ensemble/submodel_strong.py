from typing import Union

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout

from networks.tools import create_inputs, load_weights


def svhn_pyramid_ensemble_submodel_strong(input_shape=None, input_tensor=None, n_classes=None,
                                          weights_path: Union[None, str] = None) -> Model:
    """
    Defines a svhn network.

    :param n_classes: used in order to be compatible with the main script.
    :param input_shape: the input shape of the network. Can be omitted if input_tensor is used.
    :param input_tensor: the input tensor of the network. Can be omitted if input_shape is used.
    :param weights_path: a path to a trained custom network's weights.
    :return: Keras functional API Model.
    """
    inputs = create_inputs(input_shape, input_tensor)

    # Block1.
    x = Conv2D(32, (3, 3), padding='same', activation='elu', name='block1_conv1')(inputs)
    x = Conv2D(32, (3, 3), padding='same', activation='elu', name='block1_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool')(x)

    # Block2
    x = Conv2D(64, (3, 3), padding='same', activation='elu', name='block2_conv1')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='elu', name='block2_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool')(x)

    # Block3
    x = BatchNormalization(name='block3_batch-norm')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='elu', name='block3_conv')(x)
    x = Dropout(0.5, name='block3_dropout')(x)

    # Add Submodel Strong top layers.
    x = Flatten(name='flatten')(x)
    outputs = Dense(n_classes, activation='softmax', name='softmax_outputs')(x)

    # Create Submodel 1.
    model = Model(inputs, outputs, name='svhn_pyramid_ensemble_submodel_strong')
    # Load weights, if they exist.
    load_weights(weights_path, model)

    return model
