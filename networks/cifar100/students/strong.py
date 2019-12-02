from typing import Union

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation
from tensorflow.python.keras.regularizers import l2

from networks.tools import load_weights, create_inputs


def cifar100_student_strong(n_classes: int, input_shape=None, input_tensor=None,
                           weights_path: Union[None, str] = None) -> Model:
    """
    Defines a cifar100 strong student network.

    :param n_classes: the number of classes.
    :param input_shape: the input shape of the network. Can be omitted if input_tensor is used.
    :param input_tensor: the input tensor of the network. Can be omitted if input_shape is used.
    :param weights_path: a path to a trained cifar10 tiny network's weights.
    :return: Keras functional Model.
    """
    inputs = create_inputs(input_shape, input_tensor)

    # Define a weight decay for the regularisation.
    weight_decay = 1e-4

    # Block1.
    x = Conv2D(32, (3, 3), padding='same', activation='elu', name='block1_conv1',
               kernel_regularizer=l2(weight_decay))(inputs)

    x = BatchNormalization(name='block1_batch-norm1')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='elu', name='block1_conv2',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(name='block1_batch-norm2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool')(x)
    x = Dropout(0.2, name='block1_dropout')(x)

    # Block2.
    x = Conv2D(128, (3, 3), padding='same', activation='elu', name='block2_conv1',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(name='block2_batch-norm1')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='elu', name='block2_conv2',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(name='block2_batch-norm2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool')(x)
    x = Dropout(0.3, name='block2_dropout')(x)

    # Block3.
    x = Conv2D(256, (3, 3), padding='same', activation='elu', name='block3_conv1',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(name='block3_batch-norm1')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='elu', name='block3_conv2',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(name='block3_batch-norm2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block3_pool')(x)
    x = Dropout(0.4, name='block3_dropout')(x)

    # Add top layers.
    x = Flatten()(x)
    x = Dense(n_classes)(x)
    outputs = Activation('softmax', name='softmax')(x)

    # Create model.
    model = Model(inputs, outputs, name='cifar100_student_strong')

    # Load weights, if they exist.
    load_weights(weights_path, model)

    return model
