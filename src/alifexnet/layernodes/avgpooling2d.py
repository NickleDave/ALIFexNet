import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D


class AvgPool2DNode:
    def __init__(self,
                 input_size,
                 n_channels,
                 pool_size,
                 strides,
                 padding,
                 **avgpool2d_kwargs):
        """__init__ method

        Parameters
        ----------
        input_size : int
            size of one side of input (assumes square)
        n_channels : int
            number of channels in input (e.g. 3 for RGB images)
        pool_size : int
            size of one side of kernel in pooling layer (assumes square)
        strides : int
            size of strides
        padding : str
            Type of padding to use. One of {'valid', 'same', 'full'}.
        avgpool2d_kwargs : kwargs
            any other valid keyword arguments to
            tensorflow.keras.layers.AveragePooling2d
        """
        # compute shape used to reshape inputs within __call__ method
        self.shape_in = (input_size, input_size, n_channels)

        if padding == 'valid' or padding == 'same':
            # make actual layer
            self.avgpool2d = AveragePooling2D(pool_size=pool_size,
                                              strides=strides,
                                              padding=padding,
                                              **avgpool2d_kwargs)
        elif padding == 'full':
            def make_full_avgpool():
                avgpool2d = AveragePooling2D(pool_size=pool_size,
                                             strides=strides,
                                             padding='valid',
                                             **avgpool2d_kwargs)

                def full_avgpool2d(x):
                    x = tf.pad(x, pool_size - 1)
                    x = avgpool2d(x)
                    return x

                return full_avgpool2d

            self.avgpool2d = make_full_avgpool()

    def pre_build(self, shape_in, shape_out):
        pass

    def __call__(self, t, x):
        batch_size = x.get_shape()[0].value
        x = tf.reshape(x, (batch_size,) + self.shape_in)
        x = self.avgpool2d(x)
        x = tf.reshape(x, (batch_size, -1))
        return x

    def post_build(self, sess, rng):
        """loads weights and biases (if any)"""
        pass
