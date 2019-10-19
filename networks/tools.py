from tensorflow.python.keras.engine import Layer


class Crop(Layer):
    """Layer that crops (or slices) a Tensor on a given dimension from start to end."""

    def __init__(self, dimension, start, end, **kwargs):
        self.kernel = None
        self.dimension = dimension
        self.start = start
        self.end = end
        super(Crop, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Crop, self).build(input_shape)  # Be sure to call this at the end

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
