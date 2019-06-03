# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import math
import torch as to

from torch import Tensor
from typing import Dict, Tuple

import tvem
from tvem.utils import get
from tvem.utils.parallel import pprint, all_reduce
from tvem.variational.TVEMVariationalStates import TVEMVariationalStates
from tvem.variational._utils import mean_posterior
from tvem.models.TVEMModel import TVEMModel, fix_theta


class BSC(TVEMModel):
    """Binary Sparse Coding (BSC)"""

    def __init__(
        self, conf, W_init: Tensor = None, sigma_init: Tensor = None, pies_init: Tensor = None
    ):
        device = tvem.get_device()

        required_keys = ("N", "D", "H", "S", "Snew", "batch_size", "precision")
        for c in required_keys:
            assert c in conf and conf[c] is not None
        self.conf = conf
        self.required_keys = required_keys

        N, D, H, S, Snew, batch_size, precision = get(conf, *required_keys)

        self.tmp = {
            "my_Wp": to.empty((D, H), dtype=precision, device=device),
            "my_Wq": to.empty((H, H), dtype=precision, device=device),
            "my_pies": to.empty(H, dtype=precision, device=device),
            "my_sigma": to.empty(1, dtype=precision, device=device),
            "pil_bar": to.empty(H, dtype=precision, device=device),
            "WT": to.empty((H, D), dtype=precision, device=device),
            "batch_Wbar": to.empty((batch_size, S + Snew, D), dtype=precision, device=device),
            "batch_s_pjc": to.empty((batch_size, H), dtype=precision, device=device),
            "batch_Wp": to.empty((batch_size, D, H), dtype=precision, device=device),
            "batch_Wq": to.empty((batch_size, H, H), dtype=precision, device=device),
            "batch_sigma": to.empty((batch_size,), dtype=precision, device=device),
            "indS_filled": 0,
        }

        if W_init is not None:
            assert W_init.shape == (D, H)
            W_init = W_init.to(dtype=precision, device=device)
        else:
            W_init = to.rand((D, H), dtype=precision, device=device)

        if pies_init is not None:
            assert pies_init.shape == (H,)
            pies_init = pies_init.to(dtype=precision, device=device)
        else:
            pies_init = to.full((H,), 1.0 / H, dtype=precision, device=device)

        if sigma_init is not None:
            assert sigma_init.shape == (1,)
            sigma_init = sigma_init.to(dtype=precision, device=device)
        else:
            sigma_init = to.tensor([1.0], dtype=precision, device=device)

        theta = {"pies": pies_init, "W": W_init, "sigma": sigma_init}
        eps, inf = 1.0e-5, math.inf
        self.policy = {
            "W": [None, to.full_like(theta["W"], -inf), to.full_like(theta["W"], inf)],
            "pies": [
                None,
                to.full_like(theta["pies"], eps),
                to.full_like(theta["pies"], 1.0 - eps),
            ],
            "sigma": [None, to.full_like(theta["sigma"], eps), to.full_like(theta["sigma"], inf)],
        }

        super().__init__(theta=theta)

    @property
    def sorted_by_lpj(self) -> Dict[str, Tensor]:

        tmp = self.tmp

        return {"batch_Wbar": tmp["batch_Wbar"]}

    def generate_from_hidden(self, hidden_state: Tensor) -> Tensor:
        """Use hidden states to sample datapoints according to the noise model of BSC.

        :param hidden_state: a tensor with shape (N, H) where H is the number of hidden units.
        :returns: the datapoints, as a tensor with shape (N, D) where D is
                  the number of observables.
        """

        theta = self.theta

        precision, device = theta["W"].dtype, theta["W"].device
        no_datapoints, D, H_gen = (hidden_state.shape[0],) + theta["W"].shape

        Wbar = to.zeros((no_datapoints, D), dtype=precision, device=device)

        # Linear superposition
        for n in range(no_datapoints):
            for h in range(H_gen):
                if hidden_state[n, h]:
                    Wbar[n] += theta["W"][:, h]

        # Add noise according to the model parameters
        Y = Wbar + theta["sigma"] * to.randn((no_datapoints, D), dtype=precision, device=device)

        return Y

    def init_epoch(self):
        """Allocate and/or initialize tensors used during EM-step."""

        conf = self.conf
        theta = self.theta
        tmp = self.tmp

        D = conf["D"]

        tmp["my_Wp"].fill_(0.0)
        tmp["my_Wq"].fill_(0.0)
        tmp["my_pies"].fill_(0.0)
        tmp["my_sigma"].fill_(0.0)

        tmp["pil_bar"][:] = to.log(theta["pies"] / (1.0 - theta["pies"]))

        tmp["WT"][:, :] = theta["W"].t()

        tmp["pre1"] = -1.0 / 2.0 / theta["sigma"] / theta["sigma"]

        tmp["fenergy_const"] = to.log(1.0 - theta["pies"]).sum() - D / 2 * to.log(
            2 * math.pi * theta["sigma"] ** 2
        )

        tmp["infty"] = to.tensor(
            [float("inf")], dtype=theta["pies"].dtype, device=theta["pies"].device
        )

    def init_batch(self):
        """Reset counter for how many states tensors in sorted_by_lpj have been evaluated.

        Only relevant if model makes use of the sorted_by_lpj dictionary.
        """
        tmp = self.tmp
        tmp["indS_filled"] = 0

    def log_pseudo_joint(self, data: Tensor, states: Tensor) -> Tensor:
        """Evaluate log-pseudo-joints for BSC."""

        theta = self.theta
        tmp = self.tmp
        sorted_by_lpj = self.sorted_by_lpj

        batch_size, S, _ = states.shape

        pil_bar = tmp["pil_bar"]
        WT = tmp["WT"]
        pre1 = tmp["pre1"]
        indS_filled = tmp["indS_filled"]

        # TODO Find solution to avoid byte->float casting
        statesfloat = states.to(dtype=theta["W"].dtype)

        # TODO Store batch_Wbar in storage allocated at beginning of EM-step, e.g.
        # to.matmul(tensor1=states, tensor2=tmp['WT'], out=tmp["batch_Wbar"])
        sorted_by_lpj["batch_Wbar"][:batch_size, indS_filled : (indS_filled + S), :] = to.matmul(
            statesfloat, WT
        )
        batch_Wbar = sorted_by_lpj["batch_Wbar"][:batch_size, indS_filled : (indS_filled + S), :]
        tmp["indS_filled"] += S
        # is (batch_size,S)
        lpj = to.mul(to.sum(to.pow(batch_Wbar - data[:, None, :], 2), dim=2), pre1) + to.matmul(
            statesfloat, pil_bar
        )
        return lpj.to(device=states.device)

    def free_energy(self, idx: Tensor, batch: Tensor, states: TVEMVariationalStates) -> float:

        fenergy_const = self.tmp["fenergy_const"]
        lpj = states.lpj[idx]
        batch_size = lpj.shape[0]

        lpj_shifted_sum_chunk = to.logsumexp(lpj, dim=1).sum()

        return (fenergy_const * batch_size + lpj_shifted_sum_chunk).item()

    def update_param_batch(self, idx: Tensor, batch: Tensor, states: TVEMVariationalStates) -> None:

        tmp = self.tmp
        sorted_by_lpj = self.sorted_by_lpj

        lpj = states.lpj[idx]
        K = states.K[idx]
        batch_size, S, _ = K.shape

        # TODO Find solution to avoid byte->float casting
        Kfloat = K.to(dtype=lpj.dtype)

        (
            batch_s_pjc,
            batch_Wp,
            batch_Wq,
            batch_sigma,
            my_pies,
            my_Wp,
            my_Wq,
            my_sigma,
            indS_fill_upto,
            fenergy_const,
        ) = get(
            tmp,
            "batch_s_pjc",
            "batch_Wp",
            "batch_Wq",
            "batch_sigma",
            "my_pies",
            "my_Wp",
            "my_Wq",
            "my_sigma",
            "indS_fill_upto",
            "fenergy_const",
        )
        batch_Wbar = sorted_by_lpj["batch_Wbar"]

        batch_s_pjc[:batch_size, :] = mean_posterior(Kfloat, lpj)  # is (batch_size,H)
        batch_Wp[:batch_size, :, :] = batch.unsqueeze(2) * batch_s_pjc[:batch_size].unsqueeze(
            1
        )  # is (batch_size,D,H)
        batch_Wq[:batch_size, :, :] = mean_posterior(Kfloat.unsqueeze(3) * Kfloat.unsqueeze(2), lpj)
        # is (batch_size,H,H)
        batch_sigma[:batch_size] = mean_posterior(
            to.sum((batch[:, None, :] - batch_Wbar[:batch_size, :S, :]) ** 2, dim=2), lpj
        )
        # is (batch_size,)

        my_pies.add_(to.sum(batch_s_pjc[:batch_size, :], dim=0))
        my_Wp.add_(to.sum(batch_Wp[:batch_size, :, :], dim=0))
        my_Wq.add_(to.sum(batch_Wq[:batch_size, :, :], dim=0))
        my_sigma.add_(to.sum(batch_sigma[:batch_size]))

        return None

    def update_param_epoch(self) -> None:

        conf = self.conf
        theta = self.theta
        tmp = self.tmp
        policy = self.policy

        N, D = get(conf, "N", "D")
        my_pies, my_Wp, my_Wq, my_sigma = get(tmp, "my_pies", "my_Wp", "my_Wq", "my_sigma")

        theta_new = {}

        all_reduce(my_pies)
        all_reduce(my_Wp)
        all_reduce(my_Wq)
        all_reduce(my_sigma)

        # Calculate updated W
        Wold_noisy = theta["W"] + 0.1 * to.randn_like(theta["W"])
        try:
            theta_new["W"] = to.gels(my_Wp.t(), my_Wq)[0].t()
        except RuntimeError:
            pprint("Inversion error. Will not update W but add some noise instead.")
            theta_new["W"] = Wold_noisy

        # Calculate updated pi
        theta_new["pies"] = my_pies / N

        # Calculate updated sigma
        theta_new["sigma"] = to.sqrt(my_sigma / N / D)

        policy["W"][0] = Wold_noisy
        policy["pies"][0] = theta["pies"]
        policy["sigma"][0] = theta["sigma"]
        fix_theta(theta_new, policy)

        for key in theta:
            theta[key] = theta_new[key]

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.theta["W"].shape
