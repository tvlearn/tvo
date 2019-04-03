import os
import torch as to


def _default_device() -> to.device:
    dev = to.device('cpu')
    if 'TVEM_GPU' in os.environ:
        gpu_n = int(os.environ['TVEM_GPU'])
        dev = to.device(f'cuda:{gpu_n}')
    return dev


class _GlobalDevice:
    """A singleton object containing the global device settings for the framework.

    Set and get the corresponding to.device with `{set,get}_device()`.
    """
    _device: to.device = _default_device()

    @classmethod
    def set_device(cls, dev: to.device):
        cls._device = dev

    @classmethod
    def get_device(cls) -> to.device:
        return cls._device


def set_device(device: to.device):
    """Set the torch.device that all objects in the package will use by default.

    Note that certain operations might run on CPU independently of the value of tvem.device.

    The default ('cpu') can also be overridden by setting the TVEM_GPU environment variable
    to the number of the desired CUDA device. For example, in bash, `export TVEM_GPU=0`
    will make the framework default to device 'cuda:0'.
    """
    _GlobalDevice.set_device(device)


def get_device() -> to.device:
    """Get the torch.device that all objects in the package will use by default."""
    return _GlobalDevice.get_device()


def _default_run_policy() -> str:
    policy = 'seq'
    if 'TVEM_DISTRIBUTED' in os.environ and os.environ['TVEM_DISTRIBUTED'] != 0:
        policy = 'dist'
    return policy


class _GlobalPolicy:
    """A singleton object containing the global execution policy for the framework.

    Set and get the policy with `{set,get}_run_policy()`.
    """
    _policy: str = _default_run_policy()

    @classmethod
    def set_policy(cls, p: str):
        assert p in ('seq', 'dist'), "Supported policies are 'seq' and 'dist'"
        cls._policy = p

    @classmethod
    def get_policy(cls) -> str:
        return cls._policy


def set_run_policy(policy: str):
    """Set the preferred parallelization policy. Can be one of 'seq' or 'dist'.

    * 'seq': the framework will not perform any parallelization other than what torch tensors
             offer out of the box on the relevant device.
    * 'dist': the framework will perform data parallelization for the algorithms that implement it.

    The default ('seq') can also be overridden by setting the TVEM_DISTRIBUTED environment
    variable to a non-zero value.
    """
    _GlobalPolicy.set_policy(policy)


def get_run_policy() -> str:
    """Get the preferred parallelization policy for the framework.

    Returned string can be one of 'seq' and 'dist'."""
    return _GlobalPolicy.get_policy()
