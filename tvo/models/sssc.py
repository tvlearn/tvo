# -*- coding: utf-8 -*-
# Copyright (C) 2021 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import torch as to
from typing import Dict, Optional, Tuple, Union, Any
from math import pi as MATH_PI
from tvo import get_device
from tvo.utils.model_protocols import Sampler, Optimized, Reconstructor
from tvo.utils.parallel import broadcast, all_reduce, pprint
from tvo.variational.TVOVariationalStates import TVOVariationalStates
from tvo.variational._utils import mean_posterior


def _get_hash(x: to.Tensor) -> int:
    return hash(x.detach().cpu().numpy().tobytes())


class SSSC(Sampler, Optimized, Reconstructor):
    def __init__(
        self,
        H: int,
        D: int,
        W_init: to.Tensor = None,
        sigma2_init: to.Tensor = None,
        mus_init: to.Tensor = None,
        Psi_init: to.Tensor = None,
        pies_init: to.Tensor = None,
        reformulated_lpj: bool = True,
        use_storage: bool = True,
        reformulated_psi_update: bool = False,
        precision: to.dtype = to.float32,
    ):
        """Spike-And-Slab Sparse Coding (SSSC) model.

        :param H: Number of hidden units.
        :param D: Number of observables.
        :param W_init: Tensor with shape (H, D), initializes SSSC weights.
        :param sigma2_init: Tensor initializing SSSC observable variance.
        :param mus_init: Tensor with shape (H,), initializes SSSC latent means.
        :param Psi_init: Tensor with shape (H, H), initializes SSSC latent variance.
        :param pies_init: Tensor with shape (H,), initializes SSSC priors.
        :param reformulated_lpj: Use looped instead of batchified E-step and mathematically
                                 reformulated form of the log-pseudo-joint formula (exploiting
                                 matrix determinant lemma and Woodbury matrix identity). Yields
                                 more accurate solutions in large dimensions (i.e. large D and H).
        :param use_storage: Whether to memorize state vector-dependent and datapoint independent-
                            terms computed in the E-step. Terms will be looked-up rather than re-
                            computed if a datapoint evaluates a state that has been evaluated for
                            another datapoint before. The storage will be cleared after each epoch.
        :param reformulated_psi_update: Whether to update Psi using reformulated form of the
                                        update equation.
        :param precision: Floating point precision required. Must be one of torch.float32 or
                          torch.float64.
        """
        assert precision in (to.float32, to.float64), "precision must be one of torch.float{32,64}"
        device = get_device()
        self._precision = precision
        self._shape = (D, H)
        self._reformulated_lpj = reformulated_lpj
        self._use_storage = use_storage
        self._reformulated_psi_update = reformulated_psi_update

        self._theta: Dict[str, to.Tensor] = {}
        self._theta["W"] = self._init_W(W_init)
        self._theta["sigma2"] = self._init_sigma2(sigma2_init)
        self._theta["mus"] = self._init_mus(mus_init)
        self._theta["Psi"] = self._init_Psi(Psi_init)
        self._theta["pies"] = self._init_pies(pies_init)

        self._log2pi = to.log(to.tensor([2.0 * MATH_PI], dtype=precision, device=device))

        self._my_sum_y_szT = to.zeros((D, H), dtype=precision, device=device)
        self._my_sum_xpt_sz_xpt_szT = to.zeros((H, H), dtype=precision, device=device)
        self._my_sum_xpt_szszT = to.zeros((H, H), dtype=precision, device=device)
        self._my_sum_xpt_s = to.zeros((H,), dtype=precision, device=device)
        self._my_sum_xpt_sz = to.zeros((H,), dtype=precision, device=device)
        self._my_sum_xpt_ssT = to.zeros((H, H), dtype=precision, device=device)
        self._my_sum_xpt_ssz = (
            to.zeros((H, H), dtype=precision, device=device) if reformulated_psi_update else None
        )
        self._my_sum_diag_yyT = to.zeros((D,), dtype=precision, device=device)
        self._my_N = to.tensor([0], dtype=to.int, device=device)
        self._eps_eyeH = to.eye(H, dtype=precision, device=device) * 1e-6
        self._storage: Optional[Dict[int, to.Tensor]] = {} if use_storage else None

        self._config = dict(
            shape=self._shape,
            reformulated_lpj=reformulated_lpj,
            reformulated_psi_update=reformulated_psi_update,
            use_storage=use_storage,
            precision=precision,
            device=device,
        )

    def _init_W(self, init: Optional[to.Tensor]):
        D, H = self.shape
        if init is not None:
            assert init.shape == (D, H)
            return init.to(dtype=self.precision, device=get_device())
        else:
            W_init = to.rand((D, H), dtype=self.precision, device=get_device())
            broadcast(W_init)
            return W_init

    def _init_sigma2(self, init: Optional[to.Tensor]):
        if init is not None:
            assert init.shape == (1,)
            return init.to(dtype=self.precision, device=get_device())
        else:
            return to.tensor([1.0], dtype=self.precision, device=get_device())

    def _init_mus(self, init: Optional[to.Tensor]):
        H = self.shape[1]
        if init is not None:
            assert init.shape == (H,)
            return init.to(dtype=self.precision, device=get_device())
        else:
            mus_init = to.normal(
                mean=to.zeros(H, dtype=self.precision, device=get_device()),
                std=to.ones(H, dtype=self.precision, device=get_device()),
            )
            broadcast(mus_init)
            return mus_init

    def _init_Psi(self, init: Optional[to.Tensor]):
        H = self.shape[1]
        if init is not None:
            assert init.shape == (H, H)
            return init.to(dtype=self.precision, device=get_device())
        else:
            return to.eye(H, dtype=self.precision, device=get_device())

    def _init_pies(self, init: Optional[to.Tensor]):
        H = self.shape[1]
        if init is not None:
            assert init.shape == (H,)
            return init.to(dtype=self.precision, device=get_device())
        else:
            return 0.1 + 0.5 * to.rand(H, dtype=self.precision, device=get_device())

    def generate_data(
        self, N: int = None, hidden_state: to.Tensor = None
    ) -> Union[to.Tensor, Tuple[to.Tensor, to.Tensor]]:
        precision, device = self.precision, get_device()
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

        Z = to.distributions.multivariate_normal.MultivariateNormal(
            loc=self.theta["mus"],
            covariance_matrix=self.theta["Psi"],
        )

        Wbar = to.einsum("dh,nh->nd", (self.theta["W"], hidden_state * Z.sample((N,))))

        Y = to.distributions.multivariate_normal.MultivariateNormal(
            loc=Wbar,
            covariance_matrix=self.theta["sigma2"]
            * to.eye(D, dtype=self.theta["W"].dtype, device=get_device()),
        )

        return (Y.sample(), hidden_state) if must_return_hidden_state else Y.sample()

    def _lpj_fn(self, data: to.Tensor, states: to.Tensor) -> to.Tensor:
        """
        Straightforward batchified implementation of log-pseudo joint for SSSC
        """
        precision = self.precision
        W, sigma2, _pies, mus, Psi = (
            self.theta["W"],
            self.theta["sigma2"],
            self.theta["pies"],
            self.theta["mus"],
            self.theta["Psi"],
        )
        pies = _pies.clamp(1e-2, 1.0 - 1e-2)
        Kfloat = states.type_as(pies)
        N, D, S, H = data.shape + states.shape[1:]
        eyeD = to.eye(D, dtype=precision, device=get_device())

        s1 = Kfloat @ to.log(pies / (1.0 - pies))  # (N, S)
        Ws = Kfloat.unsqueeze(2) * W.unsqueeze(0).unsqueeze(1)  # (N, S, D, H)
        data_norm = data.unsqueeze(1) - Ws @ mus  # (N, S, D)
        data_norm[to.isnan(data_norm)] = 0.0

        WsPsi = Ws @ Psi  # (N, S, D, H)
        WsPsiWsT = (
            to.matmul(WsPsi, Ws.permute([0, 1, 3, 2]))
            if precision == to.float32
            else to.einsum("nsxh,nsyh->nsxy", WsPsi, Ws)
        )  # (N, S, D, D)
        C_s = WsPsiWsT + sigma2 * eyeD.unsqueeze(0).unsqueeze(1)  # (N, S, D, D)
        log_det_C_s = to.linalg.slogdet(C_s)[1]
        try:
            Inv_C_s = to.linalg.inv(C_s)
        except Exception:
            Inv_C_s = to.linalg.pinv(C_s)

        return (
            s1
            - 0.5 * log_det_C_s
            - 0.5 * (to.einsum("nsx,nsxd->nsd", data_norm, Inv_C_s) * data_norm).sum(dim=2)
        )

    def _common_e_m_step_terms(
        self, state: to.Tensor, inds_d_not_isnan: to.Tensor
    ) -> Tuple[to.Tensor, to.Tensor, to.Tensor, to.Tensor, to.Tensor, to.Tensor]:
        W = self.theta["W"]
        sigma2 = self.theta["sigma2"]
        Psi = self.theta["Psi"]
        mus = self.theta["mus"]

        W_s = W[inds_d_not_isnan][:, state]
        Psi_s = Psi[state, :][:, state]
        mus_s = mus[state]

        try:
            Inv_Psi_s = to.linalg.inv(Psi_s)
        except Exception:
            Inv_Psi_s = to.linalg.pinv(Psi_s)

        Inv_Lambda_s = W_s.t() @ W_s / sigma2 + Inv_Psi_s  # (|state|, |state|)
        try:
            Lambda_s = to.linalg.inv(Inv_Lambda_s)
        except Exception:
            Lambda_s = to.linalg.pinv(Inv_Lambda_s)

        Lambda_s_W_s_sigma2inv = Lambda_s @ W_s.t() / sigma2  # (|state|, D_nonnan)

        return (
            W_s,
            mus_s,
            Psi_s,
            Inv_Lambda_s,
            Lambda_s,
            Lambda_s_W_s_sigma2inv,
        )

    def _reformulated_lpj_fn(self, data: to.Tensor, states: to.Tensor) -> to.Tensor:
        """
        Batchified implementation of log-pseudo joint for SSSC using matrix determinant lemma and
        Woodbury matrix identity to compute determinant and inverse of matrix C_s
        """
        precision = self.precision
        sigma2, _pies = (
            self.theta["sigma2"],
            self.theta["pies"],
        )
        pies = _pies.clamp(1e-2, 1.0 - 1e-2)
        Kbool = states.to(dtype=to.bool)
        Kfloat = states.to(dtype=precision)
        batch_size, S = data.shape[0], Kbool.shape[1]

        use_storage = (
            self._use_storage if not to.isnan(data).any() else False
        )  # storage usage unrealiable for incomplete data points with differently missing entries
        if self._use_storage != use_storage:
            pprint("Disabled storage (inaccurate for incomplete data)")
        self._use_storage = use_storage

        notnan = to.logical_not(to.isnan(data))

        lpj = Kfloat @ to.log(pies / (1.0 - pies))  # initial allocation, (N, S)
        for n in range(batch_size):
            for s in range(S):
                hsh = _get_hash(Kbool[n, s])
                datapoint_notnan, D_notnan = data[n][notnan[n]], notnan[n].sum()
                if use_storage and self._storage is not None and hsh in self._storage:
                    W_s, mus_s, log_det_C_s_wo_last_term, Inv_C_s = (
                        self._storage[hsh]["W_s"],
                        self._storage[hsh]["mus_s"],
                        self._storage[hsh]["log_det_C_s_wo_last_term"],
                        self._storage[hsh]["Inv_C_s"],
                    )
                else:
                    (
                        W_s,
                        mus_s,
                        Psi_s,
                        Inv_Lambda_s,
                        Lambda_s,
                        Lambda_s_W_s_sigma2inv,
                    ) = self._common_e_m_step_terms(Kbool[n, s], notnan[n])

                    Inv_C_s = (
                        to.eye(D_notnan, dtype=self.precision, device=get_device()) / sigma2
                        - W_s @ Lambda_s_W_s_sigma2inv / sigma2
                    )  # (D_nonnan, D_nonnan)
                    log_det_C_s_wo_last_term = (
                        to.linalg.slogdet(Inv_Lambda_s)[1] + to.linalg.slogdet(Psi_s)[1]
                    )  # matrix determinant lemma, last term added in log_joint (1,)

                    if use_storage:
                        assert self._storage is not None
                        self._storage[hsh] = {
                            "W_s": W_s,
                            "mus_s": mus_s,
                            "Lambda_s": Lambda_s,
                            "Lambda_s_W_s_sigma2inv": Lambda_s_W_s_sigma2inv,
                            "log_det_C_s_wo_last_term": log_det_C_s_wo_last_term,
                            "Inv_C_s": Inv_C_s,
                        }

                datapoint_norm = datapoint_notnan - W_s @ mus_s  # (D_nonnan,)

                lpj[n, s] -= (
                    0.5
                    * (
                        log_det_C_s_wo_last_term
                        + (datapoint_norm * (Inv_C_s @ datapoint_norm)).sum()
                    ).item()
                )

        return lpj

    def log_pseudo_joint(self, data: to.Tensor, states: to.Tensor) -> to.Tensor:
        """Evaluate log-pseudo-joints for SSSC."""
        lpj_fn = self._reformulated_lpj_fn if self._reformulated_lpj else self._lpj_fn
        lpj = lpj_fn(data, states)
        min_ = to.finfo(self.precision).min
        lpj[to.isnan(lpj)] = min_
        lpj[to.isinf(lpj)] = min_
        return lpj

    def log_joint(self, data: to.Tensor, states: to.Tensor, lpj=None) -> to.Tensor:
        """Evaluate log-joints for SSSC."""
        assert states.dtype == to.uint8
        notnan = to.logical_not(to.isnan(data))
        if lpj is None:
            lpj = self.log_pseudo_joint(data, states)
        # TODO: could pre-evaluate the constant factor once per epoch
        pies = self.theta["pies"].clamp(1e-2, 1.0 - 1e-2)
        D = to.sum(notnan, dim=1)  # (N,)
        logjoints = (
            lpj
            + to.log(1.0 - pies).sum()
            - D.unsqueeze(1) / 2.0 * (self._log2pi + to.log(self.theta["sigma2"]))
        )
        assert logjoints.shape == lpj.shape
        assert not to.isnan(logjoints).any() and not to.isinf(logjoints).any()
        return logjoints

    def update_param_batch(
        self,
        idx: to.Tensor,
        batch: to.Tensor,
        states: TVOVariationalStates,
        **kwargs: Dict[str, Any],
    ) -> None:
        precision = self.precision
        lpj = states.lpj[idx]
        Kbool = states.K[idx].to(dtype=to.bool)
        Kfloat = states.K[idx].to(dtype=lpj.dtype)
        batch_size, S, H = Kbool.shape

        use_storage = self._use_storage and self._storage is not None and len(self._storage) > 0

        # TODO: Add option to neglect reconstructed values
        notnan = to.ones_like(batch, dtype=to.bool, device=batch.device)

        batch_kappas = to.zeros((batch_size, S, H), dtype=precision, device=get_device())
        batch_Lambdas_plus_kappas_kappasT = to.zeros(
            (batch_size, S, H, H), dtype=precision, device=get_device()
        )
        for n in range(batch_size):
            for s in range(S):
                state = Kbool[n, s]
                if state.sum() == 0:
                    continue
                hsh = _get_hash(state)

                datapoint = batch[n]

                if use_storage:
                    assert self._storage is not None
                    assert hsh in self._storage
                    W_s, mus_s, Lambda_s, Lambda_s_W_s_sigma2inv = (
                        self._storage[hsh]["W_s"],
                        self._storage[hsh]["mus_s"],
                        self._storage[hsh]["Lambda_s"],
                        self._storage[hsh]["Lambda_s_W_s_sigma2inv"],
                    )
                else:

                    (
                        W_s,
                        mus_s,
                        _,
                        _,
                        Lambda_s,
                        Lambda_s_W_s_sigma2inv,
                    ) = self._common_e_m_step_terms(state, notnan[n])

                datapoint_norm = datapoint - W_s @ mus_s  # (D_nonnan,)

                batch_kappas[n, s][state] = (
                    mus_s + Lambda_s_W_s_sigma2inv @ datapoint_norm
                )  # is (|state|,)
                batch_Lambdas_plus_kappas_kappasT[n, s][to.outer(state, state)] = (
                    Lambda_s + to.outer(batch_kappas[n, s][state], batch_kappas[n, s][state])
                ).flatten()  # (|state|, |state|)

        batch_xpt_s = mean_posterior(Kfloat, lpj)  # is (batch_size,H)
        batch_xpt_ssT = mean_posterior(
            Kfloat.unsqueeze(3) * Kfloat.unsqueeze(2), lpj
        )  # (batch_size, H, H)
        batch_xpt_sz = mean_posterior(batch_kappas, lpj)  # (batch_size, H)
        batch_xpt_szszT = mean_posterior(
            batch_Lambdas_plus_kappas_kappasT, lpj
        )  # (batch_size, H, H)
        batch_xpt_sszT = (
            (batch_xpt_s.unsqueeze(2) * batch_xpt_sz.unsqueeze(1))
            if self._reformulated_psi_update
            else None
        )
        # is (batch_size, H, H)

        self._my_sum_xpt_s.add_(to.sum(batch_xpt_s, dim=0))  # (H,)
        self._my_sum_xpt_ssT.add_(to.sum(batch_xpt_ssT, dim=0))  # (H, H)
        self._my_sum_xpt_sz.add_(to.sum(batch_xpt_sz, dim=0))  # (H,)
        self._my_sum_xpt_sz_xpt_szT.add_(batch_xpt_sz.t() @ batch_xpt_sz)  # (H, H)
        self._my_sum_xpt_szszT.add_(to.sum(batch_xpt_szszT, dim=0))  # (H, H)
        self._my_sum_diag_yyT.add_(to.sum(batch ** 2, dim=0))  # (D,)
        self._my_sum_y_szT.add_(batch.t() @ batch_xpt_sz)  # (D, H)
        self._my_N.add_(batch_size)  # (1,)
        if self._reformulated_psi_update:
            assert self._my_sum_xpt_ssz is not None and batch_xpt_sszT is not None
            self._my_sum_xpt_ssz.add_(to.sum(batch_xpt_sszT, dim=0))  # (H, H)

        return None

    def _invert_my_sum_xpt_ssT(self) -> to.Tensor:
        eps_eyeH = self._eps_eyeH
        try:
            Inv_my_sum_xpt_ssT = to.linalg.inv(self._my_sum_xpt_ssT)
        except Exception:
            try:
                Inv_my_sum_xpt_ssT = to.linalg.inv(self._my_sum_xpt_ssT + eps_eyeH)
                pprint("Psi update: Addd diag(eps) before computing inverse")
            except Exception:
                Inv_my_sum_xpt_ssT = to.linalg.pinv(self._my_sum_xpt_ssT + eps_eyeH)
                pprint("Psi update: Added diag(eps) and computed pseudo-inverse")
        return Inv_my_sum_xpt_ssT

    def update_param_epoch(self) -> None:
        theta = self.theta
        precision = self.precision
        device = get_device()
        dtype_eps, eps = to.finfo(precision).eps, 1e-5
        eps_eyeH = self._eps_eyeH
        D, H = self.shape

        W, sigma2, pies, Psi, mus = (
            theta["W"],
            theta["sigma2"],
            theta["pies"],
            theta["Psi"],
            theta["mus"],
        )

        all_reduce(self._my_sum_y_szT)  # (D, H)
        all_reduce(self._my_sum_xpt_szszT)  # (H, H)
        all_reduce(self._my_sum_xpt_sz_xpt_szT)  # (H, H)
        all_reduce(self._my_sum_xpt_s)  # (H,)
        all_reduce(self._my_sum_xpt_sz)  # (H,)
        all_reduce(self._my_sum_xpt_ssT)  # (H, H)
        all_reduce(self._my_sum_diag_yyT)  # (D,)
        all_reduce(self._my_N)  # (1,)

        N = self._my_N.item()

        Inv_my_sum_xpt_ssT = self._invert_my_sum_xpt_ssT()

        try:
            sum_xpt_szszT_inv = to.linalg.inv(self._my_sum_xpt_szszT)
            W[:] = self._my_sum_y_szT @ sum_xpt_szszT_inv
        except Exception:
            try:
                noise = eps * to.randn(H, dtype=precision, device=device)
                noise = noise.unsqueeze(1) * noise.unsqueeze(0)
                sum_xpt_szszT_inv = to.linalg.pinv(self._my_sum_xpt_szszT + noise)
                W[:] = self._my_sum_y_szT @ sum_xpt_szszT_inv
                pprint("W update: Used noisy pseudo-inverse")
            except Exception:
                W[:] = W + eps * to.randn_like(W)
                pprint("W update: Failed to compute W^(new). Pertubed current W with AWGN.")

        pies[:] = self._my_sum_xpt_s / N
        mus[:] = self._my_sum_xpt_sz / (self._my_sum_xpt_s + dtype_eps)
        if self._reformulated_psi_update:
            assert self._my_sum_xpt_ssz is not None
            all_reduce(self._my_sum_xpt_ssz)  # (H, H)
            _Psi = (
                to.outer(mus, mus) * self._my_sum_xpt_ssT
                + self._my_sum_xpt_szszT
                - 2.0 * mus.unsqueeze(1) * self._my_sum_xpt_ssz
            )
            Psi[:] = _Psi * Inv_my_sum_xpt_ssT + eps_eyeH
            self._my_sum_xpt_ssz[:] = 0.0
        else:
            Psi[:] = (
                self._my_sum_xpt_szszT - self._my_sum_xpt_ssT * to.outer(mus, mus)
            ) * Inv_my_sum_xpt_ssT + eps_eyeH
        sigma2[:] = (
            self._my_sum_diag_yyT.sum() - to.trace(self._my_sum_xpt_sz_xpt_szT @ (W.t() @ W))
        ) / N / D + eps

        self._my_sum_y_szT[:] = 0.0
        self._my_sum_xpt_szszT[:] = 0.0
        self._my_sum_xpt_s[:] = 0.0
        self._my_sum_xpt_sz[:] = 0.0
        self._my_sum_xpt_ssT[:] = 0.0
        self._my_sum_diag_yyT[:] = 0.0
        self._my_sum_xpt_sz_xpt_szT[:] = 0.0
        self._my_N[:] = 0.0
        if self._use_storage:
            assert self._storage is not None
            self._storage.clear()

    def data_estimator(
        self,
        idx: to.Tensor,
        batch: to.Tensor,
        states: TVOVariationalStates,
    ) -> to.Tensor:
        """Estimator used for data reconstruction. Data reconstruction can only be supported
        by a model if it implements this method. The estimator to be implemented is defined
        as follows:""" r"""
        :math:`\\langle \langle y_d \rangle_{p(y_d|\vec{s},\Theta)} \rangle_{q(\vec{s}|\mathcal{K},\Theta)}`  # noqa
        """
        # TODO Find solution to avoid redundant computations in `data_estimator` and
        # `log_pseudo_joint`
        precision = self.precision
        lpj = states.lpj[idx]
        K = states.K[idx]
        batch_size, S, H = K.shape
        Kbool = K.to(dtype=to.bool)
        W = self.theta["W"]
        use_storage = self._use_storage and self._storage is not None and len(self._storage) > 0

        notnan = to.logical_not(to.isnan(batch))

        batch_kappas = to.zeros((batch_size, S, H), dtype=precision, device=get_device())
        for n in range(batch_size):
            for s in range(S):
                state = Kbool[n, s]
                if state.sum() == 0:
                    continue
                hsh = _get_hash(state)

                datapoint_notnan = batch[n][notnan[n]]

                if use_storage:
                    assert self._storage is not None
                    assert hsh in self._storage
                    W_s, mus_s, Lambda_s_W_s_sigma2inv = (
                        self._storage[hsh]["W_s"],
                        self._storage[hsh]["mus_s"],
                        self._storage[hsh]["Lambda_s_W_s_sigma2inv"],
                    )
                else:
                    (
                        W_s,
                        mus_s,
                        _,
                        _,
                        _,
                        Lambda_s_W_s_sigma2inv,
                    ) = self._common_e_m_step_terms(state, notnan[n])

                datapoint_norm = datapoint_notnan - W_s @ mus_s  # (D_nonnan,)

                batch_kappas[n, s][state] = (
                    mus_s + Lambda_s_W_s_sigma2inv @ datapoint_norm
                )  # is (|state|,)

        return to.sum(W.unsqueeze(0) * mean_posterior(batch_kappas, lpj).unsqueeze(1), dim=2)
