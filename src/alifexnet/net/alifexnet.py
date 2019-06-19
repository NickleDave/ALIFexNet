import nengo
from nengo_dl import TensorNode

from ..layernodes import Conv2DNode

LAYERS = ['labvec',
          'conv1',
          'conv1_neuron',
          'pool1',
          'conv2',
          'conv2_neuron',
          'pool2',
          'conv3',
          'conv3_neuron',
          'conv4',
          'conv4_neuron',
          'conv5',
          'conv5_neuron',
          'pool3',
          'fc4096a',
          'fc4096a_neuron',
          'dropout1',
          'fc4096b',
          'fc4096b_neuron',
          'dropout2',
          'fc1000',
          'probs',
          'logprob']


class ALIFexNet(nengo.Network):
    def __init__(self,
                 input_size):
        with self:
            x = nengo.Node(input_size)
            x = TensorNode(Conv2DNode)
