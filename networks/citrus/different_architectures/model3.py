from typing import Union

from tensorflow.python.keras import Sequential, Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.python.keras.regularizers import l2

from networks.tools import load_weights, create_inputs


def citrus_model3(n_classes: int, input_shape=None, input_tensor=None,
                  weights_path: Union[None, str] = None) -> Sequential:
    """
    Defines a citrus network.

    :param n_classes: the number of classes.
    We use this parameter even though we know its value,
    in order to be able to use the model in order to predict some of the classes.
    :param input_shape: the input shape of the network. Can be omitted if input_tensor is used.
    :param input_tensor: the input tensor of the network. Can be omitted if input_shape is used.
    :param weights_path: a path to a trained custom network's weights.
    :return: Keras Sequential Model.
    """
    inputs = create_inputs(input_shape, input_tensor)

    # Define a weight decay for the regularisation.
    weight_decay = 1e-5

    # Block1.
    x = Conv2D(16, (2, 2), padding='same', activation='prelu', name='block1_conv1',
               kernel_regularizer=l2(weight_decay))(inputs)
    x = BatchNormalization(name='block1_batch-norm')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool')(x)
    x = Dropout(0.1, name='block1_dropout')(x)

    # Block2
    x = Conv2D(32, (3, 3), padding='same', activation='prelu', name='block2_conv1',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(name='block2_batch-norm')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool')(x)
    x = Dropout(0.2, name='block2_dropout')(x)

    # Add top layers.
    x = Flatten(name='flatten')(x)
    outputs = Dense(n_classes, activation='softmax', name='softmax_outputs')(x)

    # Create Submodel 1.
    model = Model(inputs, outputs, name='citrus_model3')
    # Load weights, if they exist.
    load_weights(weights_path, model)

    return model
