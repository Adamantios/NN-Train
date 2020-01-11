from typing import Union

from numpy.core.multiarray import ndarray
from numpy.ma import logical_or
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.python.keras.regularizers import l2

from networks.tools import create_inputs, load_weights


def cifar10_complicated_ensemble_submodel2(input_shape=None, input_tensor=None, n_classes=None,
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

    # Define a weight decay for the regularisation.
    weight_decay = 1e-4

    x = Conv2D(32, (3, 3), padding='same', activation='elu', name='conv1', kernel_regularizer=l2(weight_decay))(inputs)
    x = BatchNormalization(name='batch-norm')(x)
    x = Conv2D(32, (3, 3), padding='same', activation='elu', name='conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool')(x)
    x = Dropout(0.3, name='dropout')(x)

    # Add Submodel 2 top layers.
    x = Flatten(name='flatten')(x)
    outputs = Dense(n_classes, activation='softmax', name='softmax_outputs')(x)

    # Create Submodel 2.
    model = Model(inputs, outputs, name='cifar10_complicated_ensemble_submodel2')
    # Load weights, if they exist.
    load_weights(weights_path, model)

    return model


def cifar10_complicated_ensemble_submodel2_labels_manipulation(labels_array: ndarray) -> int:
    """
    The model's labels manipulator.

    :param labels_array: the labels to manipulate.
    :return: the number of classes predicted by the model.
    """
    labels_array[logical_or(labels_array < 1, labels_array > 3)] = 0
    return 4
