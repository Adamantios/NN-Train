from os.path import isfile
from typing import Union

from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.engine import Layer, InputSpec, InputLayer


class Crop(Layer):
    """Layer that crops (or slices) a Tensor on a given dimension from start to end."""

    def __init__(self, dimension, start, end, **kwargs):
        self.kernel = None
        self.dimension = dimension
        self.start = start
        self.end = end
        super(Crop, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(Crop, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if self.dimension == 0:
            return inputs[self.start: self.end]
        if self.dimension == 1:
            return inputs[:, self.start: self.end]
        if self.dimension == 2:
            return inputs[:, :, self.start: self.end]
        if self.dimension == 3:
            return inputs[:, :, :, self.start: self.end]
        if self.dimension == 4:
            return inputs[:, :, :, :, self.start: self.end]

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim

    def get_config(self):
        config = {
            'dimension': self.dimension,
            'start': self.start,
            'end': self.end
        }
        base_config = super(Crop, self).get_config()
        base_config.update(config)
        return base_config


def create_inputs(input_shape, input_tensor) -> Union[InputLayer, Input]:
    if input_shape is None and input_tensor is None:
        raise ValueError('You need to specify input shape or input tensor for the network.')

    # Create input.
    if input_shape is None:
        # Create an InputLayer using the input tensor.
        inputs = InputLayer(input_tensor=input_tensor, name='input')
    else:
        inputs = Input(shape=input_shape, name='input')

    return inputs


def load_weights(weights_path: str, model: Model) -> None:
    """
    Loads external weights, if the weights path exists, otherwise, raises a FileNotFoundError.

    :param weights_path: the path to the weights.
    :param model: the model.
    """
    if weights_path is not None:
        # Check if weights file exists.
        if not isfile(weights_path):
            raise FileNotFoundError('Network weights file {} does not exist.'.format(weights_path))

        # Load weights.
        model.load_weights(weights_path, True)
