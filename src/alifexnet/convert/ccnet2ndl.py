"""
convert layers from nengo_extras.cuda_convnet to config for NengoDL TensorNodes
"""
import numpy as np


CONV2D_PARAMS = ['input_size',
                 'n_channels',
                 'n_filters',
                 'kernel_size',
                 'strides',
                 'padding',
                 'weights',
                 'biases']


def conv2d(conv2d_layer):
    """converts CudaConvNet conv2d to configuration for Conv2DNode
    which can be part of a net run in a nengo-dl simulator
    using a nengo_dl.TensorNode

    Parameters
    ----------
    conv2d_layer : dict
        a conv2d layer from a "model_state" dictionary, loaded from
        a .pkl file created by Alex Krizhevsky's cuda-convnet library

    Returns
    -------
    conv2d_config : dict
        with the following keyword arguments :
            n_channels : int
            input_size : int
            n_filters : int
            kernel_size : int
            strides : int
            padding : int

    Notes
    -----
    Use the load_model_pickle function from this library to load model.pkl
    files that contain CudaConvnet configs, because that function does some
    extra clean-up
    """
    # unpack params.
    n_channels = conv2d_layer['channels']
    input_size = conv2d_layer['imgSize']
    n_filters = conv2d_layer['filters']
    kernel_size = conv2d_layer['filterSize']
    strides = conv2d_layer['stride']

    if conv2d_layer['padding'] == 0:
        padding = 'valid'
    else:
        raise ValueError('only valid (zero) padding implemented')

    # get weights + biases to load in post-build method
    weights = conv2d_layer['weights'].reshape(n_channels,
                                              kernel_size,
                                              kernel_size,
                                              n_filters)
    weights = np.rollaxis(weights, axis=-1, start=0)
    biases = conv2d_layer['biases']

    conv2d_config = dict(zip(
        CONV2D_PARAMS,
        [input_size, n_channels, n_filters, kernel_size, strides,
         padding, weights, biases]
    ))
    return conv2d_config


POOL_PARAMS = ['input_size',
               'n_channels',
               'pool_size',
               'strides',
               'padding',
               'kind']


def pool(pool_layer):
    """
    
    Parameters
    ----------
    pool_layer : dict
        a pool layer from a "model_state" dictionary, loaded from
        a .pkl file created by Alex Krizhevsky's cuda-convnet library

    Returns
    -------
    pool_config : dict
        with the following keyword arguments :
            input_size : int
            n_channels : int
            pool_size : int
            strides : int
            padding : int
            kind : str
    """
    n_channels = pool_layer['channels']
    input_size = pool_layer['imgSize']
    pool_size = pool_layer['sizeX']
    strides = pool_layer['stride']
    padding = 'full'
    kind = pool_layer['pool']

    pool_config = dict(zip(
        POOL_PARAMS,
        [input_size, n_channels, pool_size, strides, padding, kind]
    ))
    return pool_config
