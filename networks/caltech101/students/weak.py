from typing import Union

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.python.keras.regularizers import l2

from networks.tools import load_weights, create_inputs


def caltech_student_weak(n_classes: int, input_shape=None, input_tensor=None,
                         weights_path: Union[None, str] = None) -> Model:
    """
    Defines a caltech strong student network.

    :param n_classes: the number of classes.
    :param input_shape: the input shape of the network. Can be omitted if input_tensor is used.
    :param input_tensor: the input tensor of the network. Can be omitted if input_shape is used.
    :param weights_path: a path to a trained caltech tiny network's weights.
    :return: Keras functional Model.
    """
    inputs = create_inputs(input_shape, input_tensor)

    # Define a weight decay for the regularisation.
    weight_decay = 1e-4

    x = Conv2D(64, (3, 3), padding='same', activation='relu',
               input_shape=input_shape, kernel_regularizer=l2(weight_decay))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3, seed=0)(x)

    x = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4, seed=0)(x)

    x = Flatten()(x)
    x = Dense(512, kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5, seed=0)(x)
    outputs = Dense(n_classes, activation='softmax', name='softmax_outputs')(x)

    # Create model.
    model = Model(inputs, outputs, name='caltech_student_weak')
    # Load weights, if they exist.
    load_weights(weights_path, model)

    return model
