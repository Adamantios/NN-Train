from typing import Union

from numpy.core.multiarray import ndarray
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from networks.tools import create_inputs, load_weights


def cifar100_pyramid_ensemble_submodel_weak1(input_shape=None, input_tensor=None, n_classes=None,
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

    # Convolutions.
    x1 = Conv2D(128, (3, 3), padding='same', activation='elu', name='conv1')(inputs)
    x1 = Conv2D(128, (3, 3), padding='same', activation='elu', name='conv2')(x1)
    x1 = MaxPooling2D(pool_size=(2, 2), name='pool')(x1)

    # Add top layers.
    x1 = Flatten(name='flatten')(x1)
    outputs = Dense(n_classes, name='output')(x1)

    # Create Submodel 2.
    model = Model(inputs, outputs, name='cifar100_pyramid_ensemble_submodel_weak1')
    # Load weights, if they exist.
    load_weights(weights_path, model)

    return model


def cifar100_pyramid_ensemble_submodel_weak1_labels_manipulation(labels_array: ndarray) -> int:
    """
    The model's labels manipulator.

    :param labels_array: the labels to manipulate.
    :return: the number of classes predicted by the model.
    """
    labels_array[labels_array > 50] = 50
    return 51