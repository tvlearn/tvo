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


class SSSC_IN(Sampler, Optimized, Reconstructor):
    def __init__(
        self,
        H: int,
        D: int,
        W_init: to.Tensor = None,
        sigma2_init: to.Tensor = None,
        mus_init: to.Tensor = None,
        Psi_init: to.Tensor = None,
        pies_init: to.Tensor = None,
        use_storage: bool = True,
        precision: to.dtype = to.float32,
    ):
        """Spike-And-Slab Sparse Coding (SSSC_IN) model.

        :param H: Number of hidden units.
        :param D: Number of observables.
        :param W_init: Tensor with shape (H, D), initializes SSSC_IN weights.
        :param sigma2_init: Tensor initializing SSSC_IN observable variance.
        :param mus_init: Tensor with shape (H,), initializes SSSC_IN latent means.
        :param Psi_init: Tensor with shape (H, H), initializes SSSC_IN latent variance.
        :param pies_init: Tensor with shape (H,), initializes SSSC_IN priors.
        :param use_storage: Whether to memorize state vector-dependent and datapoint independent-
                            terms computed in the E-step. Terms will be looked-up rather than re-
                            computed if a datapoint evaluates a state that has been evaluated for
                            another datapoint before. The storage will be cleared after each epoch.
        :param precision: Floating point precision required. Must be one of torch.float32 or
                          torch.float64.
        """
        assert precision in (to.float32, to.float64), "precision must be one of torch.float{32,64}"
        device = get_device()
        self._precision = precision
        self._shape = (D, H)
        self._use_storage = use_storage
        
        self._theta: Dict[str, to.Tensor] = {}
        self._theta["W"] = self._init_W(W_init)
        self._theta["sigma2"] = self._init_sigma2(sigma2_init)
        self._theta["mus"] = self._init_mus(mus_init)
        self._theta["Psi"] = self._init_Psi(Psi_init)
        self._theta["pies"] = self._init_pies(pies_init)
        
        self._elbo_old = to.tensor([-to.inf])

        self._log2pi = to.log(to.tensor([2.0 * MATH_PI], dtype=precision, device=device))

        self._my_sum_y_szT = to.zeros((D, H), dtype=precision, device=device)
        self._my_sum_xpt_szszT = to.zeros((H, H), dtype=precision, device=device)
        self._my_sum_xpt_s = to.zeros((H,), dtype=precision, device=device)
        self._my_sum_xpt_z = to.zeros((H,), dtype=precision, device=device)
        self._my_sum_xpt_sz = to.zeros((H,), dtype=precision, device=device)
        self._my_sum_xpt_zzT = to.zeros((H, H), dtype=precision, device=device)
        self._my_sum_diag_yyT = to.zeros((D,), dtype=precision, device=device)
        self._my_N = to.tensor([0], dtype=to.int, device=device)
        self._my_sum_xpt_log_det_Lambda_s = to.zeros((1,), dtype=precision, device=device)
        self._my_sum_xpt_lpj = to.zeros((1,), dtype=precision, device=device)
        self._my_sum_log_sum_exp = to.zeros((1,), dtype=precision, device=device)
        self._eyeD = to.eye(D, dtype=precision, device=device)
        self._eyeH = to.eye(H, dtype=precision, device=device)
        self._eps_eyeH = to.eye(H, dtype=precision, device=device) * 1e-6
        self._storage: Optional[Dict[int, to.Tensor]] = {} if use_storage else None
        self._counter_sigma = to.zeros((1,), dtype=precision, device=device)

        self._config = dict(
            shape=self._shape,
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
            return 0.0 * init.to(dtype=self.precision, device=get_device())
        else:
            mus_init = to.normal(
                mean=to.zeros(H, dtype=self.precision, device=get_device()),
                std=to.ones(H, dtype=self.precision, device=get_device()),
            )
            broadcast(mus_init)
            return 0.0 * mus_init

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


    def _common_e_m_step_terms(
        self, state: to.Tensor
    ) -> Tuple[to.Tensor, to.Tensor, to.Tensor, to.Tensor, to.Tensor, to.Tensor]:
        K_float = state.to(dtype=self.precision)
        
        W = self.theta["W"].clone()
        sigma2 = self.theta["sigma2"].clone()
        Psi = self.theta["Psi"].clone()
        mus = self.theta["mus"].clone()

        W_s = W @ to.diag(K_float)

        try:
            Inv_Psi = to.linalg.inv(Psi)
        except Exception:
            Inv_Psi = to.linalg.pinv(Psi)

        Inv_Lambda_s = (W_s.t() @ W_s @ Psi + sigma2 * self._eyeH) @ Inv_Psi / sigma2
        
        try:
            Lambda_s = to.linalg.inv(Inv_Lambda_s)
        except Exception:
            Lambda_s = to.linalg.pinv(Inv_Lambda_s)
        
        Lambda_s_W_s_sigma2inv = Lambda_s @ W_s.t() / sigma2

        return (
            W_s,
            mus,
            Psi,
            Inv_Lambda_s,
            Lambda_s,
            Lambda_s_W_s_sigma2inv,
        )

    def _common_e_m_step_terms_notnan(
        self, states: to.Tensor
    ) -> Tuple[to.Tensor, to.Tensor, to.Tensor, to.Tensor, to.Tensor, to.Tensor]:
        Kfloat = states.to(dtype=self.precision)
        
        W, mus, Psi, sigma2 = (
            self.theta["W"].clone(),
            self.theta["mus"].clone(),
            self.theta["Psi"].clone(),
            self.theta["sigma2"].clone(),
        )
        
        W_s = W[None, None] * Kfloat[:, :, None]  # (batch_size, S, D, H)
        
        try:
            Inv_Psi = to.linalg.inv(Psi)  # (H, H)
        except Exception:
            Inv_Psi = to.linalg.pinv(Psi)  # (H, H)
        
        Inv_Lambda_s_Psi_sigma2 = W_s.transpose(-1, -2) @ W_s @ Psi[None, None]  # (batch_size, S, H, H)
        Inv_Lambda_s_Psi_sigma2 += sigma2 * self._eyeH[None, None]
        
        try:
            Lambda_s = Psi[None, None] @ to.linalg.inv(Inv_Lambda_s_Psi_sigma2)  # (batch_size, S, H, H)
        except Exception:
            Lambda_s = Psi[None, None] @ to.linalg.pinv(Inv_Lambda_s_Psi_sigma2)
        
        Lambda_s_W_s_sigma2inv = Lambda_s @ W_s.transpose(-1, -2)
        
        Lambda_s *= sigma2
        
        return (
            W_s,
            mus,
            Psi,
            Inv_Lambda_s_Psi_sigma2,
            Lambda_s,
            Lambda_s_W_s_sigma2inv,
        )

    def _check_if_storage_reliable(self, incomplete: bool, batch_size: int):
        """Disable the storage logic by setting `self._use_storage=False` if data is incomplete
        and if the specified batch_size is larger than one. The terms stored by the storage are
        only data-independent if the data does not contain missing values; otherwise,
        data-dependent indices of missing values are included in the computations of the
        respective terms.

        :param incomplete: Boolean indicating whether the data contains missing values
        :param batch_size: Batch size
        """

        use_storage = self._use_storage if not (incomplete and batch_size > 1) else False
        if self._use_storage != use_storage:
            pprint("Disabled storage (inaccurate for incomplete data and batch_size > 1)")
            self._use_storage = use_storage

    def _lpj_fn_nan(self, data: to.Tensor, states: to.Tensor) -> to.Tensor:
        precision = self.precision
        sigma2, _pies = (
            self.theta["sigma2"].clone(),
            self.theta["pies"].clone(),
        )
        pies = _pies.clamp(1e-2, 1.0 - 1e-2)
        Kbool = states.to(dtype=to.bool)
        Kfloat = states.to(dtype=precision)
        batch_size, S = data.shape[0], Kfloat.shape[1]

        self._check_if_storage_reliable(incomplete=to.isnan(data).any(), batch_size=batch_size)
        use_storage = False#self._use_storage

        lpj = Kfloat @ to.log(pies / (1.0 - pies))  # initial allocation, (N, S)
        for n in range(batch_size):
            for s in range(S):
                hsh = _get_hash(Kbool[n, s])
                datapoint = data[n]
                if use_storage:
                    raise ValueError("use_storage ist True")
                if use_storage and self._storage is not None and hsh in self._storage:
                    W_s, mus, log_det_C_s_wo_last_term, Inv_C_s = (
                        self._storage[hsh]["W_s"],
                        self._storage[hsh]["mus"],
                        self._storage[hsh]["log_det_C_s_wo_last_term"],
                        self._storage[hsh]["Inv_C_s"],
                    )
                else:
                    (
                        W_s,
                        mus,
                        Psi,
                        Inv_Lambda_s,
                        Lambda_s,
                        Lambda_s_W_s_sigma2inv,
                    ) = self._common_e_m_step_terms(Kbool[n, s])

                    Inv_C_s = (
                        self._eyeD / sigma2
                        - W_s @ Lambda_s_W_s_sigma2inv / sigma2
                    )  # (D, D)
                    log_det_C_s_wo_last_term = (
                        to.linalg.slogdet(Inv_Lambda_s)[1] + to.linalg.slogdet(Psi)[1]
                    )  # matrix determinant lemma, last term added in log_joint (1,)

                    if use_storage:
                        assert self._storage is not None
                        self._storage[hsh] = {
                            "W_s": W_s,
                            "mus": mus,
                            "Lambda_s": Lambda_s,
                            "Lambda_s_W_s_sigma2inv": Lambda_s_W_s_sigma2inv,
                            "log_det_C_s_wo_last_term": log_det_C_s_wo_last_term,
                            "Inv_C_s": Inv_C_s,
                        }

                datapoint_norm = datapoint - W_s @ mus  # (D,)

                lpj[n, s] -= (
                    0.5
                    * (
                        log_det_C_s_wo_last_term
                        + (datapoint_norm * (Inv_C_s @ datapoint_norm)).sum()
                    ).item()
                )

        return lpj

    def _lpj_fn_notnan(self, data: to.Tensor, states: to.Tensor) -> to.Tensor:
        precision = self.precision

        sigma2, _pies = (
            self.theta["sigma2"].clone(),
            self.theta["pies"].clone(),
        )
        pies = _pies.clamp(1e-2, 1.0 - 1e-2)
        Kfloat = states.to(dtype=precision)
        H = self.shape[1]

        #self._check_if_storage_reliable(incomplete=to.isnan(data).any(), batch_size=batch_size)
        use_storage = False#self._use_storage

        (
            W_s,
            mus,
            _,
            Inv_Lambda_s_Psi_sigma2,
            _,
            Lambda_s_W_s_sigma2inv,
        ) = self._common_e_m_step_terms_notnan(states=states)
        
        lpj = Kfloat @ to.log(pies / (1.0 - pies))  # initial allocation, (N, S)
        
        datapoint_norm = data[:, None] - to.einsum("nsdh,h->nsd", W_s, mus)  # (batch_size, S, D)
        
        log_det_C_s_wo_last_term = to.linalg.slogdet(Inv_Lambda_s_Psi_sigma2)[1]  # (batch_size, S)
        
        Inv_C_s_sigma2 = self._eyeD[None, None] - W_s @ Lambda_s_W_s_sigma2inv  # (batch_size, S, D, D)
        
        lpj -= 0.5 * (
            log_det_C_s_wo_last_term
            + to.einsum("nsd,nsde,nse->ns", datapoint_norm, Inv_C_s_sigma2, datapoint_norm) / sigma2
        )
        
        return lpj
    
    def _lpj_fn(self, data: to.Tensor, states: to.Tensor) -> to.Tensor:
        """
        Straightforward batchified implementation of log-pseudo joint for SSSC_HI
        """
        notnan = to.logical_not(to.isnan(data))
        
        if notnan.all():
            return self._lpj_fn_notnan(data=data, states=states)
        else:
            return self._lpj_fn_nan(data=data, states=states)
        

    def log_pseudo_joint(self, data: to.Tensor, states: to.Tensor) -> to.Tensor:
        """Evaluate log-pseudo-joints for SSSC_IN."""
        lpj = self._lpj_fn(data, states)
        min_ = to.finfo(self.precision).min
        lpj[to.isnan(lpj)] = min_
        lpj[to.isinf(lpj)] = min_
        return lpj

    def log_joint(self, data: to.Tensor, states: to.Tensor, lpj=None) -> to.Tensor:
        """Evaluate log-joints for SSSC_IN."""
        assert states.dtype == to.uint8
        
        if lpj is None:
            lpj = self.log_pseudo_joint(data, states)

        pies = self.theta["pies"]#.clamp(1e-2, 1.0 - 1e-2)
        D, H = self.shape
        
        notnan = to.logical_not(to.isnan(data))
        
        if notnan.all():
            logjoints = lpj + to.log(1.0 - pies).sum() - D / 2.0 * self._log2pi
            logjoints -= (D - H ) * 0.5 *  to.log(self.theta["sigma2"])
        else:
            logjoints = lpj + to.log(1.0 - pies).sum() - D / 2.0 * self._log2pi
            logjoints -= D / 2.0 *  to.log(self.theta["sigma2"])

        assert logjoints.shape == lpj.shape
        assert not to.isnan(logjoints).any() and not to.isinf(logjoints).any()
        
        return logjoints

    def update_param_batch_nan(
        self,
        idx: to.Tensor,
        batch: to.Tensor,
        states: TVOVariationalStates,
        **kwargs: Dict[str, Any],
    ):
        precision = self.precision
        Kbool = states.K[idx].to(dtype=to.bool)
        batch_size, S, H = Kbool.shape

        use_storage = False #self._use_storage and self._storage is not None and len(self._storage) > 0

        batch_kappas = to.zeros((batch_size, S, H), dtype=precision, device=get_device())
        batch_Lambdas_plus_kappas_kappasT = to.zeros(
            (batch_size, S, H, H), dtype=precision, device=get_device()
        )
        log_det_Lambda_s = to.zeros((batch_size, S), dtype=precision, device=get_device())
        for n in range(batch_size):
            for s in range(S):
                state = Kbool[n, s]
                hsh = _get_hash(state)

                datapoint = batch[n]

                if use_storage:
                    raise ValueError("use_storage ist True")

                if use_storage:
                    assert self._storage is not None
                    assert hsh in self._storage
                    W_s, mus, Lambda_s, Lambda_s_W_s_sigma2inv = (
                        self._storage[hsh]["W_s"],
                        self._storage[hsh]["mus"],
                        self._storage[hsh]["Lambda_s"],
                        self._storage[hsh]["Lambda_s_W_s_sigma2inv"],
                    )

                else:
                    (
                        W_s,
                        mus,
                        _,
                        _,
                        Lambda_s,
                        Lambda_s_W_s_sigma2inv,
                    ) = self._common_e_m_step_terms(state)

                log_det_Lambda_s[n,s] = to.linalg.slogdet(Lambda_s)[1]  # (1,)
                datapoint_norm = datapoint - W_s @ mus  # (D,)

                batch_kappas[n, s] = (
                    mus + Lambda_s_W_s_sigma2inv @ datapoint_norm
                )  # is (H,)
                
                batch_Lambdas_plus_kappas_kappasT[n, s] = (
                    Lambda_s + to.outer(batch_kappas[n, s], batch_kappas[n, s])
                )  # (H, H)
        
        return (
            log_det_Lambda_s,
            batch_kappas,
            batch_Lambdas_plus_kappas_kappasT,
        )
    
    def update_param_batch_notnan(
        self,
        idx: to.Tensor,
        batch: to.Tensor,
        states: TVOVariationalStates,
        **kwargs: Dict[str, Any],
    ):
        lpj = states.lpj[idx]
        Kfloat = states.K[idx].to(dtype=lpj.dtype)

        use_storage = False #self._use_storage and self._storage is not None and len(self._storage) > 0

        (
            W_s,
            mus,
            _,
            _,
            Lambda_s,
            Lambda_s_W_s_sigma2inv,
        ) = self._common_e_m_step_terms_notnan(states=Kfloat)
        
        log_det_Lambda_s = to.linalg.slogdet(Lambda_s)[1]  # (batch_size, S)
        datapoint_norm = batch[:, None] - to.einsum("nsdh,h->nsd", W_s, mus)  # (batch_size, S, D)
        
        batch_kappas = mus[None, None] + to.einsum("nshd,nsd->nsh", Lambda_s_W_s_sigma2inv, datapoint_norm)  # (batch_size, S, H)
        
        batch_Lambdas_plus_kappas_kappasT = Lambda_s + batch_kappas[:, :, :, None] @ batch_kappas[:, :, None]  # (batch_size, S, H, H)

        return (
            log_det_Lambda_s,
            batch_kappas,
            batch_Lambdas_plus_kappas_kappasT,
        )
        
    def update_param_batch(
        self,
        idx: to.Tensor,
        batch: to.Tensor,
        states: TVOVariationalStates,
        **kwargs: Dict[str, Any],
    ) -> None:
        lpj = states.lpj[idx]
        Kfloat = states.K[idx].to(dtype=lpj.dtype)
        batch_size = Kfloat.shape[0]
        
        notnan = to.logical_not(to.isnan(batch))
        
        if notnan.all():
            (
                log_det_Lambda_s,
                batch_kappas,
                batch_Lambdas_plus_kappas_kappasT,
            ) =  self.update_param_batch_notnan(
                idx=idx,
                batch=batch,
                states=states,
                **kwargs
            )
        else:
            (
                log_det_Lambda_s,
                batch_kappas,
                batch_Lambdas_plus_kappas_kappasT,
            ) = self.update_param_batch_nan(
                idx=idx,
                batch=batch,
                states=states,
                **kwargs
            )
        
        batch_xpt_s = mean_posterior(Kfloat, lpj)  # (batch_size,H)
        batch_xpt_sz = mean_posterior(Kfloat * batch_kappas, lpj)  # (batch_size, H)
        ssT = Kfloat.unsqueeze(-1) @ Kfloat.unsqueeze(-2)
        batch_xpt_szszT = mean_posterior(
            ssT * batch_Lambdas_plus_kappas_kappasT, lpj
        )  # (batch_size, H, H)
        batch_xpt_z = mean_posterior(batch_kappas, lpj)  # (batch_size, H)
        batch_xpt_zzT = mean_posterior(batch_Lambdas_plus_kappas_kappasT, lpj)  # (batch_size, H, H)
        batch_xpt_log_det_Lambda_s = mean_posterior(log_det_Lambda_s, lpj)  # (batch_size,)
        batch_xpt_lpj = mean_posterior(lpj, lpj)  # (1,)
        # stable version of log sum exp
        max_lpj = to.max(lpj, dim=-1)[0] - 10.0
        batch_log_sum_exp = to.log(to.sum(to.exp(lpj - max_lpj[:,None]), dim=-1))
        batch_log_sum_exp += max_lpj
        
        self._my_sum_xpt_s.add_(to.sum(batch_xpt_s, dim=0))  # (H,)
        self._my_sum_xpt_sz.add_(to.sum(batch_xpt_sz, dim=0))  # (H,)
        self._my_sum_xpt_szszT.add_(to.sum(batch_xpt_szszT, dim=0))  # (H, H)
        self._my_sum_xpt_z.add_(to.sum(batch_xpt_z, dim=0))  # (H,)
        self._my_sum_xpt_zzT.add_(to.sum(batch_xpt_zzT, dim=0))  # (H,H)
        self._my_sum_diag_yyT.add_(to.sum(batch**2, dim=0))  # (D,)
        self._my_sum_y_szT.add_(batch.t() @ batch_xpt_sz)  # (D, H)
        self._my_N.add_(batch_size)  # (1,)
        self._my_sum_xpt_log_det_Lambda_s.add_(to.sum(batch_xpt_log_det_Lambda_s, dim=0))  # (1,)
        self._my_sum_xpt_lpj.add_(to.sum(batch_xpt_lpj, dim=0))  # (1,)
        self._my_sum_log_sum_exp.add_(to.sum(batch_log_sum_exp, dim=0))  # (1,)

    def _compute_entropies(self, theta: Dict[str, to.Tensor]):
        # model parameters
        W, sigma2, pies, Psi, mus = (
            theta["W"].clone(),
            theta["sigma2"].clone(),
            to.clamp(theta["pies"].clone(), min=1e-7, max=1.0 - 1e-7),
            theta["Psi"].clone(),
            theta["mus"].clone(),
        )
        # expectation values
        N = self._my_N
        xpt_lpj = self._my_sum_xpt_lpj / N
        xpt_log_det_Lambda_s = self._my_sum_xpt_log_det_Lambda_s / N
        xpt_log_sum_exp = self._my_sum_log_sum_exp / N
        
        # entropy of spike variable (already checked)
        entropy_spike = - to.dot(pies, to.logit(pies)) - to.sum(to.log(1.0 - pies))
        
        # entropy of slab variable
        H = Psi.shape[-1]
        entropy_slab = H * (self._log2pi + 1.0) + to.linalg.slogdet(Psi)[1]
        entropy_slab *= 0.5
        
        # entropy of observable variable
        D = W.shape[-2]
        entropy_observable = 0.5 * D * (self._log2pi + 1.0 + to.log(sigma2))
        
        # entropy approximate posterior
        entropy_post_slab = H * (self._log2pi + 1.0) + xpt_log_det_Lambda_s
        entropy_post_slab *= 0.5
        entropy_post_spike = - xpt_lpj + xpt_log_sum_exp
        entropy_post = entropy_post_slab + entropy_post_spike
        
        return entropy_post - entropy_spike - entropy_slab - entropy_observable

    def _compute_elbo(self, theta: Dict[str, to.Tensor]):
        # model parameters
        W, sigma2, pies, Psi, mus = (
            theta["W"].clone(),
            theta["sigma2"].clone(),
            to.clamp(theta["pies"].clone(), min=1e-7, max=1.0 - 1e-7),
            theta["Psi"].clone(),
            theta["mus"].clone(),
        )
        # expectation values
        N = self._my_N
        xpt_s = self._my_sum_xpt_s / N
        xpt_z = self._my_sum_xpt_z / N
        xpt_zzT = self._my_sum_xpt_zzT / N
        xpt_y_szT = self._my_sum_y_szT / N
        xpt_szszT = self._my_sum_xpt_szszT / N
        xpt_diag_yyT = self._my_sum_diag_yyT / N
        xpt_lpj = self._my_sum_xpt_lpj / N
        xpt_log_det_Lambda_s = self._my_sum_xpt_log_det_Lambda_s / N
        xpt_log_sum_exp = self._my_sum_log_sum_exp / N
        
        
        # elbo of spike variable
        elbo_spike = to.dot(xpt_s, to.logit(pies)) + to.sum(to.log(1.0 - pies))
        
        # elbo of slab variable
        H = Psi.shape[-1]
        try:
            Inv_Psi = to.linalg.inv(Psi)
        except Exception:
            Inv_Psi = to.linalg.pinv(Psi)
        
        if not to.allclose(Psi @ Inv_Psi, self._eyeH, rtol=1e-5, atol=1e-8):
            raise ValueError("WARNING: Inverse von Psi nicht korrekt berechnet")
        
        if not to.allclose(Inv_Psi @ Psi, self._eyeH, rtol=1e-5, atol=1e-8):
            raise ValueError("WARNING: Inverse von Psi nicht korrekt berechnet")
            
        elbo_slab = -0.5 * H * self._log2pi - 0.5 * to.linalg.slogdet(Psi)[1]
        elbo_slab -= 0.5 * to.trace(Inv_Psi @ xpt_zzT)
        elbo_slab += to.dot(xpt_z, Inv_Psi @ mus)
        elbo_slab -= 0.5 * to.dot(mus, Inv_Psi @ mus)
        diff = to.trace(Inv_Psi @ xpt_zzT) - to.dot(mus, Inv_Psi @ mus) - H
        
        # elbo of observable variable
        D = W.shape[-2]
        elbo_observable = - D * (self._log2pi + to.log(sigma2))
        elbo_observable -= to.sum(xpt_diag_yyT, dim=-1, keepdim=True) / sigma2
        elbo_observable -= to.trace((W.t() @ W) @ xpt_szszT) / sigma2
        elbo_observable *= 0.5
        elbo_observable += to.trace(xpt_y_szT @ W.t()) / sigma2
        
        # elbo approximate posterior
        elbo_post_slab = H * (self._log2pi + 1.0) + xpt_log_det_Lambda_s
        elbo_post_slab *= 0.5
        elbo_post_spike = - xpt_lpj + xpt_log_sum_exp
        elbo_post = elbo_post_slab + elbo_post_spike
        
        return elbo_post + elbo_observable + elbo_spike + elbo_slab

    def update_param_epoch(self) -> None:
        theta = self.theta
        precision = self.precision
        device = get_device()
        eps = 1e-6
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
        all_reduce(self._my_sum_xpt_s)  # (H,)
        all_reduce(self._my_sum_xpt_z)  # (H,)
        all_reduce(self._my_sum_xpt_sz)  # (H,)
        all_reduce(self._my_sum_xpt_zzT)  # (H, H)
        all_reduce(self._my_sum_diag_yyT)  # (D,)
        all_reduce(self._my_N)  # (1,)
        all_reduce(self._my_sum_xpt_log_det_Lambda_s)  # (1,)
        all_reduce(self._my_sum_xpt_lpj)  # (1,)
        all_reduce(self._my_sum_log_sum_exp)  # (1,)

        N = self._my_N.item()

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
                pprint("W update: Failed to compute W^(new). Perturbed current W with AWGN.")

        pies[:] = self._my_sum_xpt_s / N
        mus[:] = self._my_sum_xpt_z / N
        Psi[:] = self._my_sum_xpt_zzT / N - to.outer(mus, mus)# + self._eps_eyeH
        
        if self._counter_sigma > 10:
            sigma2[:] = (
                self._my_sum_diag_yyT.sum() - to.trace(self._my_sum_y_szT @ W.t())
            ) / N / D# + eps
        self._counter_sigma += 1
        
        entropy_sum = self._compute_entropies(theta=theta)
        
        elbo = self._compute_elbo(theta=theta)
        
        print("Entropy sum:       ", entropy_sum)
        print("ELBO:             ", elbo)
        print("diff:               ", elbo - entropy_sum)
        
        if self._elbo_old  > elbo and not to.isclose(self._elbo_old, elbo):
            print("WARNING: M-step ELBO decreases")
            breakpoint()
        
        self._elbo_old = elbo.clone()
        
        self._my_sum_y_szT[:] = 0.0
        self._my_sum_xpt_szszT[:] = 0.0
        self._my_sum_xpt_s[:] = 0.0
        self._my_sum_xpt_z[:] = 0.0
        self._my_sum_xpt_sz[:] = 0.0
        self._my_sum_xpt_zzT[:] = 0.0
        self._my_sum_diag_yyT[:] = 0.0
        self._my_N[:] = 0.0
        self._my_sum_xpt_log_det_Lambda_s[:] = 0.0
        self._my_sum_xpt_lpj[:] = 0.0
        self._my_sum_log_sum_exp[:] = 0.0
        
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

                datapoint_norm = datapoint_notnan - W_s @ mus_s  # (D,)

                batch_kappas[n, s][state] = (
                    mus_s + Lambda_s_W_s_sigma2inv @ datapoint_norm
                )  # is (H,)

        return to.sum(W.unsqueeze(0) * mean_posterior(batch_kappas, lpj).unsqueeze(1), dim=2)
