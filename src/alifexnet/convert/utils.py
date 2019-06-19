

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