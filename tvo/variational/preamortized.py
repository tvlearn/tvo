import torch
from tvo.variational.TVOVariationalStates import TVOVariationalStates
from tvo.utils.model_protocols import Trainable, Optimized
from ._utils import update_states_for_batch, set_redundant_lpj_to_low
from tvo import get_device
import functools

class PreAmortizedVariationalStates(TVOVariationalStates):
    def __init__(
        self,
        N: int,
        H: int,
        S: int,
        model_path: str,
        nsamples: int,
        dist: str = 'posterior',
        K_init_file: str = None,
    ):
        """Truncated Variational Sampling class.

        :param N: number of datapoints
        :param H: number of latents
        :param S: number of variational states
        :param precision: floating point precision to be used for log_joint values.
                          Must be one of to.float32 or to.float64.
        :param K_init_file: Full path to H5 file providing initial states
        :param dist: either posterior or posterior_no_corr.
        """
        conf = {
            "N": N,
            "H": H,
            "S": S,
            "model_path": model_path,
            "nsamples": nsamples,
            "S_new": nsamples,
            "K_init_file": K_init_file,
            "precision": torch.float32
        }
        self.nsamples = nsamples
        self.sampler = torch.load(model_path).to(get_device())
        self.dist = dist

        self.lpj_call_count = 0

        super().__init__(conf)


    def update(self, idx: torch.Tensor, batch: torch.Tensor, model: Trainable) -> int:
        """See :func:`tvo.variational.TVOVariationalStates.update`."""

        if isinstance(model, Optimized):
            lpj_fn = model.log_pseudo_joint
            sort_by_lpj = model.sorted_by_lpj
        else:
            lpj_fn = model.log_joint
            sort_by_lpj = {}

        K, lpj = self.K, self.lpj

        lpj[idx] = lpj_fn(batch, K[idx])  # only necessary during training
        self.lpj_call_count += lpj[idx].numel()

        new_K, _ = self.sampler.sample(batch, nsamples=self.nsamples, idx=None, dist=self.dist)

        new_K = (new_K>0.5).transpose(0,1).byte()


        new_lpj = lpj_fn(batch, new_K)
        self.lpj_call_count += new_lpj.numel()

        set_redundant_lpj_to_low(new_K, new_lpj, K[idx])

        self.assertion(new_lpj)

        subs = update_states_for_batch(
            new_K, new_lpj, idx, K, lpj, sort_by_lpj=sort_by_lpj
        )

        return subs

    def assertion(self, new_lpj):
        try:
            n_lpj = new_lpj.shape.numel()
            uniques = new_lpj.unique().shape.numel()
            batch = new_lpj.shape[0]
            substituted = (new_lpj <= new_lpj.min()*0.99).sum()
            should_be_zero = n_lpj- uniques - substituted
            # print(should_be_zero)
            assert should_be_zero == 0,'{} UFOs out of {}'.format(should_be_zero, n_lpj)

        except AssertionError as e:
            # print('Not passed')
            pass

    # def make_lpj_counter(self, lpj_fn):
    #     @functools.wraps(lpj_fn)
    #     def lpj_counter(*args, **kwargs):
    #         self.lpj_call_count += 1 # criminal and heresis
    #         return lpj_fn(*args, **kwargs)
    #     return lpj_counter