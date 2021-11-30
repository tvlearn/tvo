# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0


import torch as to
import math
from torch.distributions.one_hot_categorical import OneHotCategorical

from torch import Tensor
from typing import Union, Tuple

import tvem
from tvem.utils.parallel import pprint, all_reduce, broadcast
from tvem.variational.TVEMVariationalStates import TVEMVariationalStates
from tvem.variational.fullem import FullEMSingleCauseModels
from tvem.variational._utils import mean_posterior
from tvem.utils.model_protocols import Optimized, Sampler, Reconstructor
from tvem.utils.sanity import fix_theta

# pytorch 1.2 deprecates to.gels in favour of to.lstsq
lstsq = to.lstsq if int(to.__version__.split(".")[1]) >= 2 else to.gels


class FULL_GMM(Optimized, Sampler, Reconstructor):
    def __init__(
        self,
        H: int,
        D: int,
        W_init: Tensor = None,
        Sigma_init: Tensor = None,
        pies_init: Tensor = None,
        precision: to.dtype = to.float64,
    ):
        """Full Gaussian Mixture model (FULL_GMM).

        :param H: Number of hidden units.
        :param D: Number of observables.
        :param W_init: Tensor with shape (D,H), initializes GM weights.
        :param sigma_init: Tensor with shape (D, D, H) determining the variances
        :param pies_init: Tensor with shape (H,), initializes GM priors.
        :param precision: Floating point precision required. Must be one of torch.float32 or
                          torch.float64.

        """
        assert precision in (to.float32, to.float64), "precision must be one of torch.float{32,64}"
        self._precision = precision

        device = tvem.get_device()

        if W_init is not None:
            assert W_init.shape == (D, H)
            W_init = W_init.to(dtype=precision, device=device)
        else:
            W_init = to.rand((D, H), dtype=precision, device=device)
            broadcast(W_init)

        if pies_init is not None:
            assert pies_init.shape == (H,)
            pies_init = pies_init.to(dtype=precision, device=device)
        else:
            pies_init = to.full((H,), 1.0 / H, dtype=precision, device=device)

        if Sigma_init is not None:
            assert Sigma_init.shape == (D, D, H), f"expected Sigma_intit.shape to be ({D}, {D}, {H}), got {Sigma_init.shape} instead"
            Sigma_init = Sigma_init.to(dtype=precision, device=device)
        else:
            Sigma_init = to.eye(D, dtype=precision, device=device).unsqueeze(2).repeat(1,1,D)

        self._theta = {"pies": pies_init, "W": W_init, "Sigma": Sigma_init}
        eps, inf = 1.0e-5, math.inf
        self.policy = {
            "W": [None, to.full_like(self._theta["W"], -inf), to.full_like(self._theta["W"], inf)],
            "pies": [
                None,
                to.full_like(self._theta["pies"], eps),
                to.full_like(self._theta["pies"], 1.0 - eps),
            ],
            "Sigma": [
                None,
                to.full_like(self._theta["Sigma"], -inf),
                to.full_like(self._theta["Sigma"], inf),
            ],
        }

        self.my_Wp = to.zeros((D, H), dtype=precision, device=device)
        self.my_Wq = to.zeros((H), dtype=precision, device=device)
        self.my_pies = to.zeros(H, dtype=precision, device=device)
        self.my_Sigma = to.zeros(D, D, H, dtype=precision, device=device)
        self.my_N = to.tensor([0], dtype=to.int, device=device)
        self._config = dict(H=H, D=D, precision=precision, device=device)
        self._shape = self.theta["W"].shape

    def log_pseudo_joint(self, data: Tensor, states: Tensor) -> Tensor:  # type: ignore
        """Evaluate log-pseudo-joints for full GMM."""
        # data is shape N, D
        Kfloat = states.to(
            dtype=self.theta["W"].dtype
        )  # N,C,C # TODO Find solution to avoid byte->float casting
        Wbar = to.matmul(
            Kfloat, self.theta["W"].t()
        )  # N,C,D  # TODO Pre-allocate tensor and use `out` argument of to.matmul
        # Sigma is D, D , C  -> C , D , D inverse
        try:
            lpj = to.squeeze(
                -1/2 * (Wbar - data[:, None, :]).unsqueeze(-2) 
                @ to.linalg.solve(self.theta["Sigma"].permute(2,0,1).unsqueeze(0),
                    (Wbar - data[:, None, :]).unsqueeze(-1)),
                -1
                ).squeeze(-1)
            lpj += to.matmul(Kfloat, to.log(self.theta["pies"]))

        except RuntimeError:
            lpj = to.squeeze(
                    -1/2 * (Wbar - data[:, None, :]).unsqueeze(-2) 
                    @ self.theta["Sigma"].permute(2, 1, 0).pinverse().unsqueeze(0) 
                    @ (Wbar - data[:, None, :]).unsqueeze(-1),
                    -1
                    ).squeeze(-1)
            lpj += to.matmul(Kfloat, to.log(self.theta["pies"]))
            print('lpj calculated with pinverse')

        return lpj.to(device=states.device)

    def log_joint(self, data: Tensor, states: Tensor, lpj: Tensor = None) -> Tensor:
        """Evaluate log-joints for GMM."""
        if lpj is None:
            lpj = self.log_pseudo_joint(data, states)
        D = self.shape[0]
        return lpj - D / 2 * to.log(to.tensor(2 * math.pi)) - 1 / 2 * to.log(to.det(self.theta["Sigma"].permute(2,0,1)))

    def update_param_batch(self, idx: Tensor, batch: Tensor, states: Tensor) -> None:
        lpj = states.lpj[idx]
        K = states.K[idx]
        batch_size, S, _ = K.shape

        Kfloat = K.to(dtype=lpj.dtype)  # TODO Find solution to avoid byte->float casting
        Wbar = to.matmul(
            Kfloat, self.theta["W"].t()
        )  # N,S,D # TODO Find solution to re-use evaluations from E-step

        batch_s_pjc = mean_posterior(Kfloat, lpj)  # is (batch_size,H) mean_posterior(Kfloat, lpj) 
        # basically responsibilitys: r^(n)_c
        batch_Wp = batch.unsqueeze(2) * batch_s_pjc.unsqueeze(1)  # is (batch_size,D,H)
        data_covariance = (batch[:,None,:] - Wbar).unsqueeze(3) @ (batch[:,None,:] - Wbar).unsqueeze(2) # is batch_size, S, D, D
        # batch_Sigma = to.einsum('nsde, ns->nsde',data_covariance, batch_s_pjc) # is batch_size, D, D the same as next line
        batch_Sigma = data_covariance * batch_s_pjc.unsqueeze(2).unsqueeze(2)


        # batch_Sigma = mean_posterior(to.einsum('nsd, nse-> nsde', batch[:, None, :] - Wbar, batch[:, None, :] - Wbar) , lpj)

        self.my_pies.add_(to.sum(batch_s_pjc, dim=0))
        self.my_Wp.add_(to.sum(batch_Wp, dim=0))
        self.my_Wq.add_(to.sum(batch_s_pjc, dim=0)) # sum(r(n,h),dim=0)
        self.my_Sigma.add_(to.sum(batch_Sigma, dim=0).permute(1,2,0))#my_Sigma is D,D,H; batch_Sigma is N,H,D,D
        self.my_N.add_(batch_size)

        return None

    def update_param_epoch(self) -> None:
        theta = self.theta
        policy = self.policy

        all_reduce(self.my_Wp)
        all_reduce(self.my_Wq)
        all_reduce(self.my_pies)
        all_reduce(self.my_Sigma)
        all_reduce(self.my_N)

        N = self.my_N.item()
        D = self.shape[0]

        # Calculate updated W
        Wold_noisy = theta["W"] + 0.1 * to.randn_like(theta["W"])
        broadcast(Wold_noisy)
        theta_new = {}
        try:
            theta_new["W"] = self.my_Wp / self.my_Wq[None, :]
        except RuntimeError:
            pprint("Inversion error. Will not update W but add some noise instead.")
            theta_new["W"] = Wold_noisy

        # Calculate updated pi
        theta_new["pies"] = self.my_pies / N

        # Calculate updated sigma^2
        theta_new["Sigma"] = self.my_Sigma / self.my_Wq[None,None,:] 

        policy["W"][0] = Wold_noisy
        policy["pies"][0] = theta["pies"]
        policy["Sigma"][0] = theta["Sigma"]
        fix_theta(theta_new, policy)
        for key in theta:
            theta[key] = theta_new[key]

        self.my_Wp[:] = 0.0
        self.my_Wq[:] = 0.0
        self.my_pies[:] = 0.0
        self.my_Sigma[:] = 0.0
        self.my_N[:] = 0.0

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.theta["W"].shape

    def generate_data(
        self, N: int = None, hidden_state: to.Tensor = None
    ) -> Union[to.Tensor, Tuple[to.Tensor, to.Tensor]]:
        precision, device = self.precision, tvem.get_device()
        D, H = self.shape

        if hidden_state is None:
            assert N is not None
            pies = self.theta["pies"]
            hidden_state = OneHotCategorical(probs=pies).sample([N]) == 1
            must_return_hidden_state = True
        else:
            shape = hidden_state.shape
            if N is None:
                N = shape[0]
            assert shape == (N, H), f"hidden_state has shape {shape}, expected ({N},{H})"
            must_return_hidden_state = False

        # calculate cholesky matrix L^-1 
        L_inv = to.linalg.cholesky(self.theta["Sigma"].transpose(0,2))
        Lbar = to.zeros(N, D, D, dtype=precision, device=device)

        Wbar = to.zeros((N, D), dtype=precision, device=device)
        for n in range(N):
            for h in range(H):
                if hidden_state[n, h]:
                    Wbar[n] += self.theta["W"][:, h]
                    Lbar[n] += L_inv[h,:,:]
                    

        # Add noise according to the model parameters
        eps = to.randn((N, D), dtype=precision, device=device)
        Y = Wbar + to.matmul(Lbar, eps.unsqueeze(-1)).squeeze(-1)


        return (Y, hidden_state) if must_return_hidden_state else Y

    def data_estimator(self, idx: Tensor, batch: Tensor, states: TVEMVariationalStates) -> Tensor:

        # Not yet implemented

        """Estimator used for data reconstruction. Data reconstruction can only be supported
        by a model if it implements this method. The estimator to be implemented is defined
        as follows:""" r"""
        :math:`\\langle \langle y_d \rangle_{p(y_d|\vec{s},\Theta)} \rangle_{q(\vec{s}|\mathcal{K},\Theta)}`  # noqa
        """
        # Not
        K = states.K[idx]
        # TODO Find solution to avoid byte->float casting of `K`
        # TODO Pre-allocate tensor and use `out` argument of to.matmul
        return mean_posterior(
            to.matmul(K.to(dtype=self.precision), self.theta["W"].t()), states.lpj[idx]
        )

    def pdf(self, mesh: Tensor, states: TVEMVariationalStates) -> Tensor:
        """ returns the probability density funktion of the whole model.
        :param mesh: tensor of coordinates with shape N, N , D
        :param states: TVEMVariationalStates containing all states for GMM,
        i.E. all clusters; states.K is equal to to.eye(D).tile(N,1,1)
        :returns: tensor of shape N
        """
        x_grid_points, y_grid_points, D = mesh.shape
        mesh = mesh.flatten(end_dim=1)
        lj = self.log_joint(mesh, states.K)
        return to.sum(lj.exp(), dim=1).reshape(x_grid_points, y_grid_points)
if __name__ == '__main__':
    
    model = FULL_GMM(
        1,#H
        2,#D 
        to.tensor([0,0]).unsqueeze(-1),#W 
        2 * to.eye(2).unsqueeze(-1),#sigma_init
        to.tensor(1).unsqueeze(-1),#pies_init
        )
    data = to.ones(2).unsqueeze(0)
    print(f"{data = }")
    states = to.tensor([1]).unsqueeze(-1).unsqueeze(-1)
    print(f"{states = }")
    print(model.log_joint(data,states))


    ### TESTING POSTERIOR CALCULATION ###
    N = 2
    H = 2
    D = 2
    var_states = FullEMSingleCauseModels(N,H,to.double)
    W_init = to.tensor([[0,1],[0,0]]) #W_init is D, H
    print(f'{W_init = }')
    Sigma_init = to.eye(2).unsqueeze(-1).repeat(1,1,2)#sigma_init is D, D, H
    print(f'{Sigma_init = }')
    pies_init = to.tensor([0.5,0.5]) # is (H,)
    print(f'{pies_init = }')

    model = FULL_GMM(
            H,
            D,
            W_init,
            Sigma_init,
            pies_init,
        )
    idx = to.arange(2) 
    batch = to.tensor([[0,0],[1,0]])
    var_states.update(idx, batch, model)
    lpj = var_states.lpj[idx]
    K = var_states.K[idx]
    batch_size, S, _ = K.shape

    Kfloat = K.to(dtype=lpj.dtype)  # TODO Find solution to avoid byte->float casting
    print(f'{Kfloat.shape = }')
    Wbar = to.matmul(
            Kfloat, model.theta["W"].t()
            )  # N,S,D # TODO Find solution to re-use evaluations from E-step

    batch_s_pjc = mean_posterior(Kfloat, lpj)  # is (batch_size,H) mean_posterior(Kfloat, lpj) 
    print(f'{batch_s_pjc = }')

