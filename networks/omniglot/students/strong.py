from typing import Union

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.python.keras.regularizers import l2

from networks.tools import load_weights, create_inputs


def omniglot_student_strong(n_classes: int, input_shape=None, input_tensor=None,
                            weights_path: Union[None, str] = None) -> Model:
    """
    Defines a omniglot strong student network.

    :param n_classes: the number of classes.
    :param input_shape: the input shape of the network. Can be omitted if input_tensor is used.
    :param input_tensor: the input tensor of the network. Can be omitted if input_shape is used.
    :param weights_path: a path to a trained omniglot tiny network's weights.
    :return: Keras functional Model.
    """
    inputs = create_inputs(input_shape, input_tensor)

    x = Conv2D(64, (2, 2), activation='relu', name='conv1', kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization(name='batch-norm')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool')(x)
    x = Dropout(0.3, name='dropout')(x)

    # Add top layers.
    x = Flatten(name='flatten')(x)
    outputs = Dense(n_classes, activation='softmax', name='softmax_outputs')(x)

    # Create strong student.
    model = Model(inputs, outputs, name='omniglot_student_strong')
    # Load weights, if they exist.
    load_weights(weights_path, model)

    return model
