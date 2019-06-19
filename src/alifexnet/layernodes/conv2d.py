from nengo_dl.tensor_node import reshaped

import tensorflow as tf
from tensorflow.keras.layers import Conv2D


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
        self.shape_in = (input_size, input_size, n_channels)

        # make actual layer
        self.conv2d = Conv2D(filters=filters,
                             kernel_size=kernel_size,
                             strides=strides,
                             padding=padding,
                             **conv2d_kwargs)

        # set weights/biases attributes for loading during post_build
        self.weights = weights
        self.biases = biases

    def pre_build(self, shape_in, shape_out):
        pass

    def __call__(self, t, x):
        batch_size = x.get_shape()[0].value
        x = tf.reshape(x, (batch_size,) + self.shape_in)
        x = self.conv2d(x)
        x = tf.reshape(x, (batch_size, -1))
        return x

    def post_build(self, sess, rng):
        """loads weights and biases (if any)"""
        if self.weights:
            self.conv2d.set_weights(self.weights)
        if self.biases:
            self.conv2d.set_biases(self.biases)
