"""
convert layers from nengo_extras.cuda_convnet to NengoDL TensorNodes
"""
import numpy as np

from .. import layernodes


def conv2d(conv2d_layer, **conv2d_kwargs):
    """converts CudaConvNet conv2d to an alifexnet LayerNode,
    which can be part of a net run in a nengo-dl simulator
    using a nengo_dl.TensorNode

    Parameters
    ----------
    conv2d_layer : dict
        a conv2d layer from a "model_state" dictionary, loaded from
        a .pkl file created by Alex Krizhevsky's cuda-convnet library
    conv2d_kwargs : kwargs
        any valid keyword arguments to a tensorflow.keras.layers.Conv2d
        layer that are not already defined in conv2d_layer, i.e., any
        arguments besides filters, kernel_size, strides, and padding.

    Returns
    -------
    conv2d_node : alifexnet.layernodes.Conv2DNode
        where con2d_node.conv2d will have same parameters as the
        layer from the CudaConvNet (number of filters, filter size,
        padding, etc.), and will load the weights and biases from
        that layer using its `post_build` method

    Notes
    -----
    Currently only 'valid' padding is implemented
    """
    # unpack params.
    # Note all are one-element arrays, we index to get just scalar value
    n_channels = conv2d_layer['channels'][0]
    input_size = conv2d_layer['imgSize'][0]
    n_filters = conv2d_layer['filters']
    kernel_size = conv2d_layer['filterSize'][0]
    strides = conv2d_layer['stride'][0]
    padding = conv2d_layer['padding'][0]
    if padding == 0:
        padding = 'valid'
    else:
        raise NotImplementedError('only valid (zero) padding implemented')

    # get weights + biases to load in post-build method
    weights = conv2d_layer['weights'][0].reshape(n_channels,
                                                 kernel_size,
                                                 kernel_size,
                                                 n_filters)
    weights = np.rollaxis(weights, axis=-1, start=0)
    biases = conv2d_layer['biases']

    return layernodes.Conv2DNode(input_size,
                                 n_channels,
                                 filters=n_filters,
                                 kernel_size=kernel_size,
                                 strides=strides,
                                 padding=padding,
                                 weights=weights,
                                 biases=biases,
                                 **conv2d_kwargs)
