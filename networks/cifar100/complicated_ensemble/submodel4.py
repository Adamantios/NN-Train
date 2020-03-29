from typing import Union

from numpy.core.multiarray import ndarray
from numpy.ma import logical_or
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.python.keras.regularizers import l2

from networks.tools import create_inputs, load_weights


def cifar100_complicated_ensemble_submodel4(input_shape=None, input_tensor=None, n_classes=None,
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

    # Define a weight decay for the regularisation.
    weight_decay = 1e-4

    # Block 1.
    x = Conv2D(32, (3, 3), padding='same', activation='relu', name='block1_conv1', kernel_regularizer=l2(weight_decay))(
        inputs)
    x = BatchNormalization(name='block1_batch-norm1')(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu', name='block1_conv2', kernel_regularizer=l2(weight_decay))(
        x)
    x = BatchNormalization(name='block1_batch-norm2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool')(x)
    x = Dropout(0.2, name='block1_dropout', seed=0)(x)

    # Block 2.
    x = Conv2D(64, (3, 3), padding='same', activation='relu', name='block2_conv1', kernel_regularizer=l2(weight_decay))(
        x)
    x = BatchNormalization(name='block2_batch-norm1')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu', name='block2_conv2', kernel_regularizer=l2(weight_decay))(
        x)
    x = BatchNormalization(name='block2_batch-norm2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool')(x)
    x = Dropout(0.3, name='block2_dropout', seed=0)(x)

    # Block 3.
    x = Conv2D(128, (3, 3), padding='same', activation='relu', name='block3_conv1',
               kernel_regularizer=l2(weight_decay))(
        x)
    x = BatchNormalization(name='block3_batch-norm1')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu', name='block3_conv2',
               kernel_regularizer=l2(weight_decay))(
        x)
    x = BatchNormalization(name='block3_batch-norm2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block3_pool')(x)
    x = Dropout(0.4, name='block3_dropout', seed=0)(x)

    # Block 4.
    x = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_conv1',
               kernel_regularizer=l2(weight_decay))(
        x)
    x = BatchNormalization(name='block4_batch-norm1')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_conv2',
               kernel_regularizer=l2(weight_decay))(
        x)
    x = BatchNormalization(name='block4_batch-norm2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block4_pool')(x)
    x = Dropout(0.5, name='block4_dropout', seed=0)(x)

    # Add top layers.
    x = Flatten(name='flatten')(x)
    outputs = Dense(n_classes, activation='softmax', name='softmax_outputs')(x)

    # Create Submodel 4.
    model = Model(inputs, outputs, name='cifar100_complicated_ensemble_submodel4')
    # Load weights, if they exist.
    load_weights(weights_path, model)

    return model


def cifar100_complicated_ensemble_submodel4_labels_manipulation(labels_array: ndarray) -> int:
    """
    The model's labels manipulator.

    :param labels_array: the labels to manipulate.
    :return: the number of classes predicted by the model.
    """
    labels_array[logical_or(labels_array < 60, labels_array > 79)] = -1

    for i, label_i in enumerate(range(60, 80)):
        labels_array[labels_array == label_i] = i + 1

    labels_array[labels_array == -1] = 0

    return 21
