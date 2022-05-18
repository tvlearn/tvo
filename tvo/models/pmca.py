# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import math
import torch as to

from torch import Tensor
from typing import Union, Tuple

import tvo
from tvo.utils.parallel import pprint, all_reduce, broadcast
from tvo.variational.TVOVariationalStates import TVOVariationalStates
from tvo.variational._utils import mean_posterior, lpj2pjc
from tvo.utils.model_protocols import Optimized, Sampler, Reconstructor
from tvo.utils.sanity import fix_theta

torch_minor_version = int(to.__version__.split(".")[1])
if torch_minor_version >= 10:
    # pytorch 1.10 deprecates to.lstsq in favour of to.linalg.lstsq,
    # which takes arguments in reversed order
    def lstsq(a, b):
        return to.linalg.lstsq(b, a)

elif torch_minor_version >= 2:
    # pytorch 1.2 deprecates to.gels in favour of to.lstsq
    lstsq = to.lstsq
else:
    lstsq = to.gels


class PMCA(Optimized, Sampler, Reconstructor):
    def __init__(
        self,
        H: int,
        D: int,
        W_init: Tensor = None,
        #sigma2_init: Tensor = None,
        pies_init: Tensor = None,
        individual_priors: bool = True,
        precision: to.dtype = to.float64,
    ):
        """Shallow Binary Sparse Coding (BSC) model.

        :param H: Number of hidden units.
        :param D: Number of observables.
        :param W_init: Tensor with shape (D,H), initializes BSC weights.
        :param pies_init: Tensor with shape (H,), initializes BSC priors.
        :param individual_priors: Whether to use a Bernoulli prior with H individual prior
         probabilities. If False, the same prior probability will be used for all latents.
        :param precision: Floating point precision required. Must be one of torch.float32 or
                          torch.float64.

        """
        assert precision in (to.float32, to.float64), "precision must be one of torch.float{32,64}"
        self._precision = precision

        device = tvo.get_device()

        if W_init is not None:
            assert W_init.shape == (D, H)
            W_init = W_init.to(dtype=precision, device=device)
        else:
            W_init = to.rand((D, H), dtype=precision, device=device)
            broadcast(W_init)

        if pies_init is not None:
            assert pies_init.shape == (H,) if individual_priors else pies_init.shape == (1,)
            pies_init = pies_init.to(dtype=precision, device=device)
        else:
            pies_init = (
                to.full((H,), 1.0 / H, dtype=precision, device=device)
                if individual_priors
                else to.tensor([1.0 / H], dtype=precision, device=device)
            )

        #if sigma2_init is not None:
        #    assert sigma2_init.shape == (1,)
        #    sigma2_init = sigma2_init.to(dtype=precision, device=device)
        #else:
        #    sigma2_init = to.tensor([1.0], dtype=precision, device=device)

        self._theta = {"pies": pies_init, "W": W_init}
        eps, inf = 1.0e-5, math.inf
        self.policy = {
            "W": [None, to.full_like(self._theta["W"], eps), to.full_like(self._theta["W"], inf)],
            "pies": [
                None,
                to.full_like(self._theta["pies"], eps),
                to.full_like(self._theta["pies"], 1.0 - eps),
            ],
            #"sigma2": [
            #    None,
            #    to.full_like(self._theta["sigma2"], eps),
            #    to.full_like(self._theta["sigma2"], inf),
            #],
        }

        self.my_Wp = to.zeros((D, H), dtype=precision, device=device)
        self.my_Wq = to.zeros((D, H), dtype=precision, device=device)
        self.my_pies = to.zeros(H, dtype=precision, device=device)
        #self.my_sigma2 = to.zeros(1, dtype=precision, device=device)
        self.my_N = to.tensor([0], dtype=to.int, device=device)
        self._config = dict(
            H=H, D=D, individual_priors=individual_priors, precision=precision, device=device
        )
        self._shape = self.theta["W"].shape

    def log_pseudo_joint(self, data: Tensor, states: Tensor) -> Tensor:  # type: ignore
        """Evaluate log-pseudo-joints for PMCA"""
        Kfloat = states.to(
            dtype=self.theta["W"].dtype
        )  # TODO Find solution to avoid byte->float casting
        #Wbar = to.matmul(
        #    Kfloat, self.theta["W"].t()
        #)  # TODO Pre-allocate tensor and use `out` argument of to.matmul
        N, no_states = Kfloat.size()[:2]
        #breakpoint()
        D = self.theta["W"].size()[0]
        Wbar = to.zeros(N, no_states, D)
        #for s in range(no_states):
        #    for n in range(N):
        #        Wbar[n, s, :] = to.amax(self.theta["W"] * Kfloat[n, s, :], 1)
        #    #Wbar[s, :] = to.amax(self.theta["W"] * Kfloat[0, s, :], 1)   # TODO: dounle-check the 0 index!!

        # sanity check for avoiding zero values!
        #Wbar = to.maximum(Wbar, 1e-4 + to.zeros(N, no_states, D))
        #breakpoint()

        Kpriorterm = (
            to.matmul(Kfloat, to.log(self.theta["pies"] / (1 - self.theta["pies"])))
            if self.config["individual_priors"]
            else to.log(self.theta["pies"] / (1 - self.theta["pies"])) * Kfloat.sum(dim=2)
        )

        #fac_approx = this_y[this_x_infr]
        #N_num = np.shape(this_y[this_x_infr])[0]
        #breakpoint()
        #factorial_term = to.zeros(N)
        #index = to.where(data>0)
        #breakpoint()
        #factorial_term[index] = (data[index] * to.log(data[index]) - data[index]+(to.log(data[index]*(1+4*data[index]*(1+2*data[index])))/6.)+(to.log(to.tensor(math.pi))/2.))
        #lpjpt = to.nansum(data[:, None] * to.log(Wbar.t()) - Wbar.t() - factorial_term[:, None], dim=1) # is (S,) 

        #lpj = to.matmul(data, to.log(Wbar.t())) - to.nansum(Wbar, 1)[None, :] + Kpriorterm    # is (N, no_states) 

        lpj = to.zeros(N, no_states, dtype=self._precision)
        for n in range(N):
            my_y = data[n, :]
            index = to.where(my_y>0)
            factorial_term = my_y[index] * to.log(my_y[index]) - my_y[index]

            Wbar_term = to.amax(self.theta["W"][:, None, :] * Kfloat[n, :, :], 2).t()
            Wbar[n, :, :] = to.maximum(Wbar_term, 1e-4 + to.zeros(no_states, D))
            lpj[n, :] = to.matmul(data[n, :], to.log(Wbar[n, :, :].t())) - to.nansum(Wbar[n, :, :], 1) - to.nansum(factorial_term)[None] + Kpriorterm[n, :]
            #lpj[n, :] = to.matmul(data[n, :], to.log(Wbar[n, :, :].t())) - to.nansum(Wbar[n, :, :], 1) + Kpriorterm[n, :]

        # LINEAR MODEL:
        #lpj = (
        #    to.mul(
        #        to.nansum(to.pow(Wbar - data[:, None, :], 2), dim=2), -1 / 2 / self.theta["sigma2"]
        #    )
        #    + Kpriorterm
        #)
        return lpj.to(device=states.device)

    def log_joint(self, data: Tensor, states: Tensor, lpj: Tensor = None) -> Tensor:
        """Evaluate log-joints for BSC."""
        if lpj is None:
            lpj = self.log_pseudo_joint(data, states)
        D = to.sum(to.logical_not(to.isnan(data)), dim=1)  # (N,)
        H = self.shape[1]
        priorterm = (
            to.log(1 - self.theta["pies"]).sum()
            if self.config["individual_priors"]
            else H * to.log(1 - self.theta["pies"])
        )
        return lpj + priorterm # - D.unsqueeze(1) / 2 * to.log(2 * math.pi * self.theta["sigma2"])

    def update_param_batch(self, idx: Tensor, batch: Tensor, states: TVOVariationalStates) -> None:
        lpj = states.lpj[idx]
        K = states.K[idx]
        D, H = self.theta["W"].size()
        batch_size, S, _ = K.shape
        Kfloat = K.to(dtype=lpj.dtype)  # TODO Find solution to avoid byte->float casting
        #Wbar = to.matmul(
        #    Kfloat, self.theta["W"].t()
        #)  # TODO Find solution to re-use evaluations from E-step


        batch_s_pjc = mean_posterior(Kfloat, lpj)  # is (batch_size,H)
        #batch_Wp = batch.unsqueeze(2) * batch_s_pjc.unsqueeze(1)  # is (batch_size,D,H)
        #Kq = Kfloat.mul(lpj2pjc(lpj)[:, :, None])
        #batch_Wq = to.einsum("ijk,ijl->kl", Kq, Kfloat)  # is (batch_size,H,H)

        #my_W = to.broadcast_to(self.theta["W"], (batch_size, D, H)) # is (batch_size, D, H)
        #to.argmax(my_W * Kfloat[:, 0, :], 1)

        Adh = to.zeros(batch_size, S, D, H)
        for s in range(S):
            for n in range(batch_size):
                Adh_term = to.argmax(self.theta["W"] * Kfloat[n, s, :], 1)
                for d in range(D):
                    Adh[n, s, d, Adh_term[d]] = 1

        #for n in range(batch_size):
        #    Adh_term = to.argmax(self.theta["W"][:, None, :] * Kfloat[n, :, :], 2)
        #    for d in range(D):
        #        Adh[n, :, d, Adh_term[:,d]] = 1
        
        batch_Adh_pjc = mean_posterior(Adh, lpj) # is (batch_size, D, H)
        batch_Wp = batch_Adh_pjc * batch[:, :, None]

        self.my_pies.add_(to.sum(batch_s_pjc, dim=0))
        self.my_Wp.add_(to.sum(batch_Wp, dim=0))
        self.my_Wq.add_(to.sum(batch_Adh_pjc, dim=0))
        #self.my_Wq.add_(batch_Wq)
        #self.my_sigma2.add_(to.sum(batch_sigma2))
        self.my_N.add_(batch_size)
        #breakpoint()

        return None

    def update_param_epoch(self) -> None:
        theta = self.theta
        policy = self.policy

        all_reduce(self.my_Wp)
        all_reduce(self.my_Wq)
        all_reduce(self.my_pies)
        all_reduce(self.my_N)

        N = self.my_N.item()
        D, H = self.shape

        # Calculate updated W
        Wold_noisy = theta["W"] + 0.1 * to.randn_like(theta["W"])
        dim1, dim2 = Wold_noisy.size()
        Wold_noisy = to.maximum(Wold_noisy, 1e-4 + to.zeros(dim1, dim2))
        broadcast(Wold_noisy)
        theta_new = {}
        index = to.where(self.my_Wq<1e-4)
        my_Wq_corrected = self.my_Wq
        my_Wq_corrected[index] = 1e-4

        try:
            theta_new["W"] = self.my_Wp / my_Wq_corrected
        except RuntimeError:
            pprint("Inversion error. Will not update W but add some noise instead.")
            theta_new["W"] = Wold_noisy

        # Calculate updated pi
        theta_new["pies"] = (
            self.my_pies / N
            if self.config["individual_priors"]
            else self.my_pies.sum(dim=0, keepdim=True) / N / H
        )

        policy["W"][0] = Wold_noisy
        policy["pies"][0] = theta["pies"]
        fix_theta(theta_new, policy)
        for key in theta:
            theta[key][:] = theta_new[key]

        self.my_Wp[:] = 0.0
        self.my_Wq[:] = 0.0
        self.my_pies[:] = 0.0
        self.my_N[:] = 0.0

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.theta["W"].shape

    def generate_data(
        self, N: int = None, hidden_state: to.Tensor = None
    ) -> Union[to.Tensor, Tuple[to.Tensor, to.Tensor]]:
        precision, device = self.precision, tvo.get_device()
        D, H = self.shape

        if hidden_state is None:
            assert N is not None
            pies = self.theta["pies"]
            hidden_state = to.rand((N, H), dtype=precision, device=device) < pies
            must_return_hidden_state = True
        else:
            shape = hidden_state.shape
            if N is None:
                N = shape[0]
            assert shape == (N, H), f"hidden_state has shape {shape}, expected ({N},{H})"
            must_return_hidden_state = False


        Y = to.zeros(N, D)
        Wbar = to.zeros((N, D), dtype=precision, device=device)

        # Non-linear superposition
        for n in range(N):
            for h in range(H):
                if hidden_state[n, h]:
                    Wbar[n] = to.maximum(Wbar[n], self.theta["W"][:, h])  

            Wbar_ = to.maximum(Wbar[n], 1e-4 + to.zeros(D))          

            # Add noise according to the model parameters   
            Y[n] = to.poisson(Wbar_)

        ## Linear superposition 
        #for n in range(N):
        #    for h in range(H):
        #        if hidden_state[n, h]:
        #            Wbar[n] += self.theta["W"][:, h]

        ## Add noise according to the model parameters
        #Y = Wbar + to.sqrt(self.theta["sigma2"]) * to.randn((N, D), dtype=precision, device=device)

        return (Y, hidden_state) if must_return_hidden_state else Y

    def data_estimator(self, idx: Tensor, batch: Tensor, states: TVOVariationalStates) -> Tensor:
        """Estimator used for data reconstruction. Data reconstruction can only be supported
        by a model if it implements this method. The estimator to be implemented is defined
        as follows:""" r"""
        :math:`\\langle \langle y_d \rangle_{p(y_d|\vec{s},\Theta)} \rangle_{q(\vec{s}|\mathcal{K},\Theta)}`  # noqa
        """
        K = states.K[idx]
        # TODO Find solution to avoid byte->float casting of `K`
        # TODO Pre-allocate tensor and use `out` argument of to.matmul
        N_, no_states_ = K.size()[:2]
        D = self.theta["W"].size()[0]
        Wbar = to.zeros(N_, no_states_, D)
        #for s in range(no_states_):
        #    for n in range(N_):
        #        Wbar[n, s, :] = to.amax(self.theta["W"] * K[n, s, :], 1)
        #    #Wbar[s, :] = to.amax(self.theta["W"] * Kfloat[0, s, :], 1)   # TODO: dounle-check the 0 index!!

        for n in range(N_):
            Wbar[n, :, :] = to.amax(self.theta["W"][:, None, :] * K[n, :, :], 2).t()

        # sanity check for avoiding zero values!
        Wbar = to.maximum(Wbar, 1e-4 + to.zeros(N_, no_states_, D))

        return mean_posterior(Wbar, states.lpj[idx])
        #return mean_posterior(
        #    to.matmul(K.to(dtype=self.precision), self.theta["W"].t()), states.lpj[idx]
        #)

