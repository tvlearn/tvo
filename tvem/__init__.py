import torch as _to
import os as _os

device: _to.device = _to.device('cpu')
"""The torch.device that all objects in the package will use by default.

    Note that certain operations might run on CPU independently of
    the value of tvem.device.

    To change the default, simply assign a new torch.device to tvem.device.
"""
if 'TVEM_USE_GPU' in _os.environ and _os.environ['TVEM_USE_GPU'] != 0:
    device = _to.device('cuda:0')
