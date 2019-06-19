import os
from pathlib import Path

import nengo_extras.data
from nengo_extras.cuda_convnet import load_model_pickle


REPO_LOCAL_PATH = Path(os.getcwd())

# layers, weights, etc.
REPO_LOCAL_DATA_PATH = REPO_LOCAL_PATH.joinpath('data')
VIS_NET_DATA_FNAME = 'ilsvrc2012-lif-48.pkl'
VIS_NET_DATA_PATH = REPO_LOCAL_DATA_PATH.joinpath(
    'neural_net_weights/{}'.format(VIS_NET_DATA_FNAME)
)
VIS_NET_DATA_PATH = str(VIS_NET_DATA_PATH)

VIS_NET_DATA_URL = 'https://ndownloader.figshare.com/files/5370917?private_link=f343c68df647e675af28'

# download if not already
VIS_NET_DATA_PATH = nengo_extras.data.get_file(filename=VIS_NET_DATA_PATH,
                                               url=VIS_NET_DATA_URL)
VIS_NET_DATA = load_model_pickle(str(VIS_NET_DATA_PATH))
