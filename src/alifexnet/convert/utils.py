"""
adapted from https://github.com/nengo/nengo-extras
"""
import os
import pickle

import numpy as np


def pickle_load(file, *args, **kwargs):
    kwargs.setdefault('encoding', 'latin1')
    return pickle.load(file, *args, **kwargs)


def load_model_pickle(loadfile):
    loadfile = os.path.expanduser(loadfile)
    with open(loadfile, 'rb') as f:
        model = pickle_load(f)

    def _scalar_or_arr(val):
        """return scalar value if val is a one-element array,
        else return array"""
        if type(val) == list:
            if len(val) == 1:
                return val[0]
            else:
                return val
        else:
            return val

    new_layers = {}
    for layer_name, layer_config in model['model_state']['layers'].items():
        new_layers[layer_name] = {
            key: _scalar_or_arr(val)
            for key, val in layer_config.items()
        }

    model['model_state']['layers'] = new_layers
    return model


def layer_depths(layers):
    """determine depth of layer in network

    Parameters
    ----------
    layers : dict
        where key is layer name and value is dictionary of metadata
    """
    depths = {}

    def get_depth(name):
        """recursively iterate through a layers' inputs and inputs to those 
        inputs, etc., to determine the first layer's level in a stack of 
        layers"""
        # if this layer is already in depths, just return its depth
        if name in depths:
            return depths[name]

        # return this layer's inputs, or an empty list
        inputs = layers[name].get('inputs', [])
        # apply get_depth to inputs and find max (or return 0 if no inputs)
        depth = max(get_depth(i) for i in inputs) + 1 if len(inputs) > 0 else 0
        depths[name] = depth
        return depth

    for name in layers:
        get_depth(name)

    return depths