from typing import Union

from numpy.core.multiarray import ndarray
from numpy.ma import logical_or
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.python.keras.regularizers import l2

from networks.tools import create_inputs, load_weights


def omniglot_complicated_ensemble_submodel2(input_shape=None, input_tensor=None, n_classes=None,
                                            weights_path: Union[None, str] = None) -> Model:
    """
    Defines a omniglot network.

    :param n_classes: the number of classes.
    :param input_shape: the input shape of the network. Can be omitted if input_tensor is used.
    :param input_tensor: the input tensor of the network. Can be omitted if input_shape is used.
    :param weights_path: a path to a trained custom network's weights.
    :return: Keras functional API Model.
    """
    inputs = create_inputs(input_shape, input_tensor)

    # Define a weight decay for the regularisation.
    weight_decay = 5e-4

    # Block1.
    x = Conv2D(32, (3, 3), activation='elu', name='block1_conv1',
               kernel_regularizer=l2(weight_decay))(inputs)
    x = BatchNormalization(name='block1_batch-norm')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool')(x)
    x = Dropout(0.2, name='block1_dropout')(x)

    # Block2
    x = Conv2D(64, (3, 3), activation='elu', name='block2_conv1',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(name='block2_batch-norm')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool')(x)
    x = Dropout(0.4, name='block2_dropout')(x)

    # Add top layers.
    x = Flatten(name='flatten')(x)
    outputs = Dense(n_classes, activation='softmax', name='softmax_outputs')(x)

    # Create Submodel 1.
    model = Model(inputs, outputs, name='omniglot_complicated_ensemble_submodel2')
    # Load weights, if they exist.
    load_weights(weights_path, model)

    return model


def omniglot_complicated_ensemble_submodel2_labels_manipulation(labels_array: ndarray) -> int:
    """
    The model's labels manipulator.

    :param labels_array: the labels to manipulate.
    :return: the number of classes predicted by the model.
    """
    # Labels [324 - 648].
    labels_array[logical_or(labels_array < 324, labels_array > 648)] = -1

    for i, label_i in enumerate(range(324, 649)):
        labels_array[labels_array == label_i] = i

    labels_array[labels_array == -1] = 325

    return 326
