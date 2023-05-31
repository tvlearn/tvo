import torch
from tvo.variational.TVOVariationalStates import TVOVariationalStates
from tvo.utils.model_protocols import Trainable, Optimized
from ._utils import update_states_for_batch, set_redundant_lpj_to_low, _unique_ind
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

        B, S, H = K[idx].shape

        lpj[idx] = lpj_fn(batch, K[idx])  # only necessary during training
        self.lpj_call_count += lpj[idx].numel()

        new_K, _ = self.sampler.sample(batch, nsamples=self.nsamples, idx=None, dist=self.dist)

        new_K = (new_K>0.5).transpose(0,1).byte()

        new_K = self.make_unique(torch.concat((new_K,K[idx]), dim=1), nbatch=len(idx))

        new_lpj = lpj_fn(batch, new_K)

        values, indices = new_lpj.sort()

        self.lpj_call_count += new_lpj.numel()

        self.lpj[idx]=values[:,-S:]
        for n in range(B):
            self.K[idx[n]] = new_K[n, indices[n, -S:]]

        # set_redundant_lpj_to_low(new_K, new_lpj, K[idx])

        # self.debug_lpj(new_lpj)

        # subs = update_states_for_batch(
        #     new_K, new_lpj, idx, K, lpj, sort_by_lpj=sort_by_lpj
        # )

        return 0

    def debug_lpj(self, new_lpj):
        n_lpj = new_lpj.shape.numel()
        uniques = new_lpj.unique().shape.numel()
        batch = new_lpj.shape[0]
        substituted = (new_lpj <= new_lpj.min()*0.99).sum()
        should_be_zero = n_lpj- uniques - substituted
        # print('{} lpj out of {} are equal to others'.format(should_be_zero, n_lpj))


    def make_unique(self, new_K, nbatch):
        min_len=self.nsamples
        for n in range(nbatch):
            keep_ind = _unique_ind(new_K[n])
            new_K[n,0:len(keep_ind)]= new_K[n][keep_ind]
            if min_len>len(keep_ind):
                min_len = len(keep_ind)
        new_k = new_K[:, :min_len]
        # print('Lowest amount of unique states={}'.format(min_len))
        return new_k

    # def make_lpj_counter(self, lpj_fn):
    #     @functools.wraps(lpj_fn)
    #     def lpj_counter(*args, **kwargs):
    #         self.lpj_call_count += 1 # criminal and heresis
    #         return lpj_fn(*args, **kwargs)
    #     return lpj_counter