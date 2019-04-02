import torch as _to
import os as _os

device: _to.device = _to.device('cpu')
"""The torch.device that all objects in the package will use by default.

    Note that certain operations might run on CPU independently of
    the value of tvem.device.

    The default ('cpu') can be overridden by exporting the TVEM_GPU
    environment variable with a non-zero value. At runtime, simply assign
    a new torch.device to tvem.device to change the framework's behavior.
"""
if 'TVEM_GPU' in _os.environ and _os.environ['TVEM_GPU'] != 0:
    device = _to.device('cuda:0')


class Policy:
    def __init__(self, policy: str):
        """Execution policy.

        :param policy: supported policies are `'seq'` (for sequential execution)
                       and `'dist'` (for distributed execution).
        """
        assert policy in ('seq', 'dist'), 'Supported policies are "seq" and "dist"'
        self.policy = policy

    def __repr__(self):
        return f"Policy('{self.policy}')"


policy: Policy = Policy('seq')
"""The preferred parallelization policy. Can be one of Policy('seq') or Poicy('dist').

    * 'seq': the framework will not perform any parallelization other
      than what torch tensors offer out of the box on the relevant device.
    * 'dist': the framework will perform data parallelization
      for the algorithms that implement it.

    The default ('seq') can be overridden by exporting the TVEM_DISTRIBUTED
    environment variable with a non-zero value. At runtime, simply assign
    a new value to tvem.policy to change the framework's behavior.
"""
if 'TVEM_DISTRIBUTED' in _os.environ and _os.environ['TVEM_DISTRIBUTED'] != 0:
    policy = Policy('dist')
