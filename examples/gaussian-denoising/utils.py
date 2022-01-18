import os
import sys
import h5py
import imageio
import numpy as np
import torch as to
import torch.distributed as dist
from typing import Tuple
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from typing import Dict, Union

from tvo import get_run_policy
from tvo.utils.parallel import init_processes as _init_processes


def init_processes() -> Tuple[int, int]:
    if get_run_policy() == "seq":
        return 0, 1
    else:
        assert get_run_policy() == "mpi"
        _init_processes()
        return dist.get_rank(), dist.get_world_size()


class stdout_logger(object):
    """Redirect print statements both to console and file

    Source: https://stackoverflow.com/a/14906787
    """

    def __init__(self, txt_file):
        self.terminal = sys.stdout
        self.log = open(txt_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def get_image(image_file: str, rescale: float = None) -> to.Tensor:
    """Read image from file, optionally rescale image size and return as to.Tensor

    :param image_file: Full path to image file (.png, .jpg, ...)
    :param rescale: If provided, the image height and width will be rescaled by this factor
    :return: Image as torch tensor
    """
    img = imageio.imread(image_file)
    isrgb = np.ndim(img) == 3 and img.shape[2] == 3
    isgrey = np.ndim(img) == 2
    assert isrgb or isgrey, "Expect img image to be either RGB or grey"
    if rescale is not None:
        orig_shape = img.shape
        target_shape = [int(orig_shape[1] * rescale), int(orig_shape[0] * rescale)]
        img = (
            np.asarray(
                [
                    np.asarray(
                        Image.fromarray(img[:, :, ch]).resize(target_shape, resample=Image.NEAREST),
                        dtype=np.float64,
                    )
                    for ch in range(3)
                ]
            ).transpose(1, 2, 0)
            if isrgb
            else np.asarray(
                Image.fromarray(img).resize(target_shape, resample=Image.NEAREST), dtype=np.float64
            )
        )
        print("Resized input image from {}->{}".format(orig_shape, np.asarray(img).shape))
        return to.from_numpy(img)
    else:
        return to.from_numpy(np.asarray(img, dtype=np.float64))


def store_as_h5(to_store_dict: Dict[str, to.Tensor], output_name: str) -> None:
    """Takes dictionary of tensors and writes to H5 file

    :param to_store_dict: Dictionary of torch Tensors
    :param output_name: Full path of H5 file to write data to
    """
    os.makedirs(os.path.split(output_name)[0], exist_ok=True)
    with h5py.File(output_name, "w") as f:
        for key, val in to_store_dict.items():
            f.create_dataset(key, data=val if isinstance(val, float) else val.detach().cpu())
    print(f"Wrote {output_name}")


def get_epochs_from_every(every: int, total: int) -> to.Tensor:
    """Return indices corresponding to every Xth. Sequence starts at (every - 1) and always
    includes (total - 1) as last step.

    :param every: Step interval
    :param total: Total number of steps
    :return: Step indices

    Example:
    >>> print(get_epochs_from_every(2, 9))
    >>>
    """
    return to.unique(
        to.cat((to.arange(start=every - 1, end=total, step=every), to.tensor([total - 1])))
    )


def eval_fn(
    target: Union[np.ndarray, to.Tensor], reco: Union[np.ndarray, to.Tensor], data_range: int = 255
) -> to.Tensor:
    return to.tensor(
        peak_signal_noise_ratio(
            target.detach().cpu().numpy() if isinstance(target, to.Tensor) else target,
            reco.detach().cpu().numpy() if isinstance(reco, to.Tensor) else reco,
            data_range=data_range,
        )
    )
