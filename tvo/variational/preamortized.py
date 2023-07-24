import torch
from tvo.variational.TVOVariationalStates import TVOVariationalStates
from tvo.utils.model_protocols import Trainable, Optimized
from ._utils import update_states_for_batch, set_redundant_lpj_to_low
from tvo import get_device


class PreAmortizedVariationalStates(TVOVariationalStates):
    def __init__(
        self,
        N: int,
        H: int,
        S: int,
        model_path: str,
        nsamples: int,
        K_init_file: str = None,
        use_corr: bool = True,
    ):
        """Truncated Variational Sampling class.

        :param N: number of datapoints
        :param H: number of latents
        :param S: number of variational states
        :param precision: floating point precision to be used for log_joint values.
                          Must be one of to.float32 or to.float64.
        :param K_init_file: Full path to H5 file providing initial states
        :param use_corr: Whether to use correlations or not
        """
        conf = {
            "N": N,
            "H": H,
            "S": S,
            "model_path": model_path,
            "nsamples": nsamples,
            "S_new": nsamples,
            "K_init_file": K_init_file,
            "precision": torch.float32,
            "use_corr": use_corr,
        }
        self.nsamples = nsamples
        self.sampler = torch.load(model_path).to(get_device())
        self.dist = "posterior" if use_corr else "posterior_no_corr"
        self.lpj_call_count = 0

        super().__init__(conf)

    def update(self, idx: torch.Tensor, batch: torch.Tensor, model: Trainable) -> int:

        if isinstance(model, Optimized):
            lpj_fn = model.log_pseudo_joint
            sort_by_lpj = model.sorted_by_lpj
        else:
            lpj_fn = model.log_joint
            sort_by_lpj = {}

        K, lpj = self.K, self.lpj

        B, S, H = K[idx].shape

        lpj[idx] = lpj_fn(batch, K[idx])
        self.lpj_call_count += lpj[idx].numel()

        with torch.no_grad:
            new_K, _ = self.sampler.sample(batch, nsamples=self.nsamples, idx=None, dist=self.dist)

        new_K = (new_K > 0.5).transpose(0, 1).byte()

        new_K = self.make_unique(new_K, B, S)

        new_lpj = lpj_fn(batch, new_K)

        set_redundant_lpj_to_low(new_K, new_lpj, K[idx])

        subs = update_states_for_batch(new_K, new_lpj, idx, K, lpj, sort_by_lpj=sort_by_lpj)

        return subs

    # def debug_lpj(self, new_lpj):
    #     n_lpj = new_lpj.shape.numel()
    #     uniques = new_lpj.unique().shape.numel()
    #     batch = new_lpj.shape[0]
    #     substituted = (new_lpj <= new_lpj.min()*0.99).sum()
    #     should_be_zero = n_lpj- uniques - substituted
    #     # print('{} lpj out of {} are equal to others'.format(should_be_zero, n_lpj))
    #
    def make_unique(self, new_K, nbatch, S):
        # assert new_K.shape[1] > S
        min_len = self.nsamples
        to = torch
        for n in range(nbatch):
            uniques = new_K[n].unique(dim=0)
            new_K[n, 0 : len(uniques)] = uniques

            if min_len > len(uniques):
                min_len = len(uniques)
        new_k = new_K[:, :min_len]
        # print('Lowest amount of unique states={}'.format(min_len))
        return new_k

    #
    # def check_joint_k_and_knew_are_unique(self,new_lpj, new_K, idx, S, B):
    #     for b, i in enumerate(idx):
    #         nui = torch.where(new_lpj[b] > torch.finfo(torch.float32).min)[0]
    #         new_K_unique = new_K[b][nui]
    #         maybe_unique_K = torch.cat((self.K[i], new_K_unique))
    #         possibly_unique_K = self.make_unique(maybe_unique_K, B, S)
    #         assert possibly_unique_K.shape[0] == (nui.shape[0] + S), 'Disparity of uniques in concat[k,knew]'
    #
    # # def make_lpj_counter(self, lpj_fn):
    # #     @functools.wraps(lpj_fn)
    # #     def lpj_counter(*args, **kwargs):
    # #         self.lpj_call_count += 1 # criminal and heresis
    # #         return lpj_fn(*args, **kwargs)
    # #     return lpj_counter
