from typing import Union

from numpy.core.multiarray import ndarray
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.python.keras.regularizers import l2

from networks.tools import create_inputs, load_weights


def caltech_complicated_ensemble_submodel1(input_shape=None, input_tensor=None, n_classes=None,
                                           weights_path: Union[None, str] = None) -> Model:
    """
    Defines a caltech network.

    :param n_classes: used in order to be compatible with the main script.
    :param input_shape: the input shape of the network. Can be omitted if input_tensor is used.
    :param input_tensor: the input tensor of the network. Can be omitted if input_shape is used.
    :param weights_path: a path to a trained custom network's weights.
    :return: Keras functional API Model.
    """
    inputs = create_inputs(input_shape, input_tensor)

    # Define a weight decay for the regularisation.
    weight_decay = 1e-4

    x = Conv2D(64, (3, 3), padding='same', activation='relu',
               input_shape=input_shape, kernel_regularizer=l2(weight_decay))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(x)

    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(x)

    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(x)

    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(x)

    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(x)

    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(x)

    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(x)

    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(x)

    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(x)

    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(x)

    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(x)

    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(x)

    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Dense(512, kernel_regularizer=l2(weight_decay))(x)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)
    outputs = Dense(n_classes, activation='softmax', name='softmax_outputs')(x)

    # Create Submodel 1.
    model = Model(inputs, outputs, name='caltech_complicated_ensemble_submodel1')
    # Load weights, if they exist.
    load_weights(weights_path, model)

    return model


def caltech_complicated_ensemble_submodel1_labels_manipulation(labels_array: ndarray) -> int:
    """
    The model's labels manipulator.

    :param labels_array: the labels to manipulate.
    :return: the number of classes predicted by the model.
    """
    labels_array[labels_array > 22] = 22
    return 23
