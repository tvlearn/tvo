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
    ):
        """Truncated Variational Sampling class.

        :param N: number of datapoints
        :param H: number of latents
        :param S: number of variational states
        :param precision: floating point precision to be used for log_joint values.
                          Must be one of to.float32 or to.float64.
        :param K_init_file: Full path to H5 file providing initial states
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

        lpj[idx] = lpj_fn(batch, K[idx])

        # batch_size, H = batch.shape[0], K.shape[2]

        # new_K_prior = (
        #     torch.rand(batch_size, self.config["S_new_prior"], H, device=K.device)
        #     < model.theta["pies"]
        # ).byte()

        # approximate_marginals = (
        #     mean_posterior(K[idx].type_as(lpj), lpj[idx])
        #     .unsqueeze(1)
        #     .expand(batch_size, self.config["S_new_marg"], H)
        # )  # approximates p(s_h=1|\yVecN, \Theta), shape is (batch_size, S_new_marg, H)
        # new_K_marg = (
        #     to.rand(batch_size, self.config["S_new_marg"], H, device=K.device)
        #     < approximate_marginals
        # ).byte()

        # new_K = to.cat((new_K_prior, new_K_marg), dim=1)

        new_K, _ = self.sampler.sample(batch, nsamples=self.nsamples, idx=None, dist='posterior')
        # new_K, _ = self.sampler.sample(batch, nsamples=self.nsamples, idx=None, dist='full_marginal')

        new_K = (new_K>0.5).transpose(0,1).byte()
        new_lpj = lpj_fn(batch, new_K)

        set_redundant_lpj_to_low(new_K, new_lpj, K[idx])

        return update_states_for_batch(
            new_K, new_lpj, idx, K, lpj, sort_by_lpj=sort_by_lpj
        )
