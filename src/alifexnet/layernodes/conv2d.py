import tensorflow as tf
from tensorflow.keras.layers import Conv2d


class Conv2DNode:
    def __init__(self,
                 input_size,
                 n_channels,
                 filters,
                 kernel_size,
                 strides,
                 padding,
                 weights=None,
                 biases=None,
                 **conv2d_kwargs):
        """__init__ method

        Parameters
        ----------
        input_size : int
            size of one side of input (assumes square)
        n_channels : int
            number of channels in input (e.g. 3 for RGB images)
        filters : int
             number of filters in Conv2D layer
        kernel_size : int
            size of one side of kernels in Conv2D layer (assumes square)
        strides : int
            size of strides in Conv2D layer
        padding : int
            size of padding. Currently only accepts 0, i.e., the 'valid'
            option for padding in keras.layers.Conv2D
        conv2d_kwargs : kwargs
            any other valid keyword argument to tensorflow.keras.layers.Conv2D
        """
        # compute shape used to reshape inputs within __call__ method
        image_shape = (input_size, input_size, n_channels)
        self.input_shape = (-1,) + image_shape

        # make actual layer
        self.conv2d = Conv2D(filters=filters,
                             kernel_size=kernel_size,
                             strides=strides,
                             padding=padding,
                             **conv2d_kwargs)

        # make weights/biases attribute for loading during post_build
        self.weights = weights
        self.biases = biases

    def pre_build(self, shape_in, shape_out):
        pass

    def __call__(self, t, x):
        # reshape flattened inputs into 2D shape
        # (plus batch dimension)
        inputs = tf.reshape(x, )
        return self.conv2d(images)

    def post_build(self, sess, rng):
        """loads weights and biases (if any)"""
        if self.weights:
            self.conv2d.set_weights(self.weights)
        if self.biases:
            self.conv2d.set_biases(self.biases)
