from os.path import isfile
from typing import Union

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, InputLayer, Dense, Flatten


def cifar100_model3(n_classes: int, input_shape=None, input_tensor=None,
                    weights_path: Union[None, str] = None) -> Sequential:
    """
    Defines a cifar10 network.

    :param n_classes: the number of classes.
    :param input_shape: the input shape of the network. Can be omitted if input_tensor is used.
    :param input_tensor: the input tensor of the network. Can be omitted if input_shape is used.
    :param weights_path: a path to a trained custom network's weights.
    :return: Keras Sequential Model.
    """
    if input_shape is None and input_tensor is None:
        raise ValueError('You need to specify input shape or input tensor for the network.')

    # Create a Sequential model.
    model = Sequential(name='cifar100_model3')

    if input_shape is None:
        # Create an InputLayer using the input tensor.
        model.add(InputLayer(input_tensor=input_tensor))

    # Block1
    if input_tensor is None:
        first_conv = Conv2D(64, (3, 3), padding='same', activation='elu', name='block1_conv1', input_shape=input_shape)

    else:
        first_conv = Conv2D(64, (3, 3), padding='same', activation='elu', name='block1_conv1')

    model.add(first_conv)
    model.add(Conv2D(64, (3, 3), padding='same', activation='elu', name='block1_conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='block1_pool'))

    # Block2
    model.add(Conv2D(128, (3, 3), padding='same', activation='elu', name='block2_conv1'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='elu', name='block2_conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='block2_pool'))

    # Block3
    model.add(Conv2D(256, (3, 3), padding='same', activation='elu', name='block3_conv1'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='elu', name='block3_conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='block3_pool'))

    # Add top layers.
    model.add(Flatten())
    model.add(Dense(n_classes, activation='softmax'))

    if weights_path is not None:
        # Check if weights file exists.
        if not isfile(weights_path):
            raise FileNotFoundError('Network weights file {} does not exist.'.format(weights_path))

        # Load weights.
        model.load_weights(weights_path, True)

    return model
