from typing import Union

from numpy.core.multiarray import ndarray
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.python.keras.regularizers import l2

from networks.tools import create_inputs, load_weights


def svhn_complicated_ensemble_submodel5(input_shape=None, input_tensor=None, n_classes=None,
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

    # Define a weight decay for the regularisation.
    weight_decay = 1e-4

    # Block1.
    x = Conv2D(32, (3, 3), padding='same', activation='elu', name='block1_conv1',
               kernel_regularizer=l2(weight_decay))(inputs)
    x = BatchNormalization(name='block1_batch-norm')(x)
    x = Conv2D(32, (3, 3), padding='same', activation='elu', name='block1_conv2',
               kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool')(x)
    x = Dropout(0.2, name='block1_dropout', seed=0)(x)

    # Block2
    x = Conv2D(32, (3, 3), padding='same', activation='elu', name='block3_conv1',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(name='block2_batch-norm')(x)
    x = Conv2D(32, (3, 3), padding='same', activation='elu', name='block3_conv2',
               kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block3_pool')(x)
    x = Dropout(0.4, name='block2_dropout', seed=0)(x)

    # Add top layers.
    x = Flatten(name='flatten')(x)
    outputs = Dense(n_classes, activation='softmax', name='softmax_outputs')(x)

    # Create Submodel 5.
    model = Model(inputs, outputs, name='svhn_complicated_ensemble_submodel5')
    # Load weights, if they exist.
    load_weights(weights_path, model)

    return model


def svhn_complicated_ensemble_submodel5_labels_manipulation(labels_array: ndarray) -> int:
    """
    The model's labels manipulator.

    :param labels_array: the labels to manipulate.
    :return: the number of classes predicted by the model.
    """
    labels_array[labels_array < 8] = 0
    labels_array[labels_array == 8] = 1
    labels_array[labels_array == 9] = 2
    return 3
