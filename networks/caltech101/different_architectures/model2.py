from typing import Union

from tensorflow.python.keras import Sequential, Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, \
    Dense, Flatten
from tensorflow.python.keras.regularizers import l2

from networks.tools import load_weights, create_inputs


def caltech_model2(n_classes: int, input_shape=None, input_tensor=None,
                   weights_path: Union[None, str] = None) -> Sequential:
    """
    Defines a caltech network.

    :param n_classes: the number of classes.
    We use this parameter even though we know its value,
    in order to be able to use the model in order to predict some of the classes.
    :param input_shape: the input shape of the network. Can be omitted if input_tensor is used.
    :param input_tensor: the input tensor of the network. Can be omitted if input_shape is used.
    :param weights_path: a path to a trained custom network's weights.
    :return: Keras Sequential Model.
    """
    inputs = create_inputs(input_shape, input_tensor)

    # Define a weight decay for the regularisation.
    weight_decay = 5e-4

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

    x = Flatten()(x)
    x = Dense(512, kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(n_classes, activation='softmax', name='softmax_outputs')(x)

    # Create model.
    model = Model(inputs, outputs, name='caltech_model2')
    # Load weights, if they exist.
    load_weights(weights_path, model)

    return model
