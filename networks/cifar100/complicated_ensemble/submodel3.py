from typing import Union

from numpy.core.multiarray import ndarray
from numpy.ma import logical_or
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from networks.tools import create_inputs, load_weights


def cifar100_complicated_ensemble_submodel3(input_shape=None, input_tensor=None, n_classes=None,
                                            weights_path: Union[None, str] = None) -> Model:
    """
    Defines a cifar100 network.

    :param n_classes: used in order to be compatible with the main script.
    :param input_shape: the input shape of the network. Can be omitted if input_tensor is used.
    :param input_tensor: the input tensor of the network. Can be omitted if input_shape is used.
    :param weights_path: a path to a trained custom network's weights.
    :return: Keras functional API Model.
    """
    inputs = create_inputs(input_shape, input_tensor)

    # Block1.
    x = Conv2D(128, (3, 3), padding='same', activation='elu', name='block1_conv1')(inputs)
    x = Conv2D(128, (3, 3), padding='same', activation='elu', name='block1_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool')(x)

    # Block2
    x = Conv2D(256, (3, 3), padding='same', activation='elu', name='block2_conv1')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='elu', name='block2_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool')(x)

    # Add top layers.
    x = Flatten(name='flatten')(x)
    outputs = Dense(n_classes, activation='softmax', name='softmax_outputs')(x)

    # Create Submodel 3.
    model = Model(inputs, outputs, name='cifar100_complicated_ensemble_submodel3')
    # Load weights, if they exist.
    load_weights(weights_path, model)

    return model


def cifar100_complicated_ensemble_submodel3_labels_manipulation(labels_array: ndarray) -> int:
    """
    The model's labels manipulator.

    :param labels_array: the labels to manipulate.
    :return: the number of classes predicted by the model.
    """
    labels_array[logical_or(labels_array < 30, labels_array > 59)] = -1

    for i, label_i in enumerate(range(30, 60)):
        labels_array[labels_array == label_i] = i

    labels_array[labels_array == -1] = 30

    return 31
