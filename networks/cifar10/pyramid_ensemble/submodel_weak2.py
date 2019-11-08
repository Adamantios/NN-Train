from typing import Union

from numpy.core.multiarray import ndarray
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from networks.tools import create_inputs, load_weights


def cifar10_pyramid_ensemble_submodel_weak2(input_shape=None, input_tensor=None, n_classes=None,
                                            weights_path: Union[None, str] = None) -> Model:
    """
    Defines a cifar10 network.

    :param n_classes: used in order to be compatible with the main script.
    :param input_shape: the input shape of the network. Can be omitted if input_tensor is used.
    :param input_tensor: the input tensor of the network. Can be omitted if input_shape is used.
    :param weights_path: a path to a trained custom network's weights.
    :return: Keras functional API Model.
    """
    inputs = create_inputs(input_shape, input_tensor)

    # Convolutions.
    x = Conv2D(64, (3, 3), padding='same', activation='elu', name='conv1')(inputs)
    x = Conv2D(64, (3, 3), padding='same', activation='elu', name='conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool')(x)

    # Add top layers.
    x = Flatten(name='flatten')(x)
    outputs = Dense(n_classes, activation='softmax', name='softmax_outputs')(x)

    # Create Submodel 2.
    model = Model(inputs, outputs, name='cifar10_pyramid_ensemble_submodel_weak2')
    # Load weights, if they exist.
    load_weights(weights_path, model)

    return model


def cifar10_pyramid_ensemble_submodel_weak2_labels_manipulation(labels_array: ndarray) -> int:
    """
    The model's labels manipulator.

    :param labels_array: the labels to manipulate.
    :return: the number of classes predicted by the model.
    """
    labels_array[labels_array < 5] = 0
    labels_array[labels_array == 5] = 1
    labels_array[labels_array == 6] = 2
    labels_array[labels_array == 7] = 3
    labels_array[labels_array == 8] = 4
    labels_array[labels_array == 9] = 5
    return 6
