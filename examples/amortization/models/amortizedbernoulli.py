import math
from enum import Enum
import torch
import torch.nn as nn
from torch.nn import Module, Parameter
from .common import *


def compute_probabilities(log_p_joint):
    """ Returns vectors probabilities (normalized)
        :param log_f        : [N, K]    log-joint probability of (X, Kset) 
    """
    Ps = log_p_joint.exp() / (log_p_joint.exp().sum(-1).unsqueeze(-1))
    return Ps


def compute_marginal(Kset, log_p_joint):
    """ Returns bits marginal probability
        :param Kset         : [N, K, H] truncated binary posterior sets
        :param log_p_joint  : [N, K]    log-joint probability of (X, Kset) 
    """
    Ps = compute_probabilities(log_p_joint)
    marginal = Kset * Ps[..., None]
    marginal = marginal.sum(-2)
    return marginal


def binarize(x, threshold=0.5):
    return (x > threshold).float()


def sample_mean(x, weights=None, dim=0):
    """ Returns sample mean
        :param x                : [N, D]
        :param weights          : [N]       probabilities of data points
        :returns mean           : [D]
    """
    if weights is not None:
        return torch.sum(x * weights.unsqueeze(-1), dim=dim)
    return torch.mean(x, dim=dim)


def sample_covar(x, weights=None):
    """ Returns sample covariance
        :param x                : [N, D]
        :param weights          : [N]
        :returns covar          : [D, D]
    """
    if weights is not None:
        x_mean = sample_mean(x, weights)
        x_zeromean = x - x_mean
        x_weighted = torch.sqrt(weights).unsqueeze(-1) * x_zeromean
        x_cov = torch.mm(x_weighted.T, x_weighted)
        return x_cov
    return torch.cov(x.T, aweights=weights, correction=0)


def batch_sample_covar(x, weights=None):
    """ Returns M-batch sample covariance along 0 dimension
        :param x                : [M, N, D]
        :param weights          : [M, N]
        :returns covar          : [M, D, D]
    """
    if weights is None:
        return torch.stack([sample_covar(xi) for xi in x])
    else:
        return torch.stack([sample_covar(x[i], weights[i]) for i in range(x.shape[0])])


def stable_entropy(log_f):
    """ Computes entropy from unnormalized log-probability (energy)
        :param log_f    : [N, K]    K energy values (batch of N distributions)
    """
    log_f = log_f - log_f.max(dim=-1, keepdim=True)[0]
    Z = torch.exp(log_f).sum(-1)
    H = torch.log(Z) - 1/Z * (torch.exp(log_f) * log_f).sum(-1) 
    return H

class SamplerModule(Module):
    def sample_q(self, X, indexes=None, nsamples=1000):
        raise NotImplementedError()


Objective = Enum("Objective", ["CROSSENTROPY", "KLDIVERGENCE", "MEANKLDIVERGENCE"])


class AmortizedBernoulli(SamplerModule):
    def __init__(self, nsamples=10, variationalparams=None) -> None:
        super().__init__()
        self.nsamples = nsamples
        self.variationalparams = variationalparams
        self.temperature = 1.0
        self.objective_type = Objective.KLDIVERGENCE


    def forward(self, X, Kset, log_p, marginal_p=None, indexes=None):
        """ Returns cross-entropy H(p_K | q(X)) and KL-divergence(p_K || q(X))
            :param X            : [N, D]    N data points
            :param Kset         : [N, K, H] truncated binary posterior sets
            :param log_p        : [N, K]    log-joint probability of (X, Kset) 
            :param marginal_p   : [N, H]    marginal probability of bits
            :param indexes      : [N]       data points indexes
        """
        return self.importance_sampling_objective(X, Kset, log_p, marginal_p, indexes)
    

    def naive_relaxed_objective(self, X, Kset, log_p, marginal_p=None, indexes=None):
        """ Returns relaxed cross-entropy H(p_K | q(X))
            :param X            : [N, D]    N data points
            :param Kset         : [N, K, H] truncated binary posterior sets
            :param log_p        : [N, K]    log-joint probability of (X, Kset) 
            :param marginal_p   : [N, H]    marginal probability of bits
            :param indexes      : [N]       data points indexes
        """
        N, K, H = Kset.shape
        assert Kset.shape[0] == X.shape[0]
        assert Kset.shape[:2] == log_p.shape

        device = X.device

        if marginal_p is None:
            marginal_p = compute_marginal(Kset, log_p)
            marginal_p = torch.clamp(marginal_p, min=0.001, max=0.999)
        
        # Create training samples of relaxed Kset
        relaxed_mu = -torch.log((1-marginal_p)/(marginal_p)).unsqueeze(-2)  # inverse logistic of marginal
        relaxed_Kset_samples, relaxed_log_p = RandomFlowSequence(
            source=RelaxedBernoulliSource(marginal=marginal_p, 
                                          Kset=Kset, 
                                          device=device), 
            sequence=[
                ElementwiseLinearTransform(0.999, 0.0005),  # for stability
                CollapsedBernoulliLogisticRelaxation(mu=relaxed_mu, t=self.temperature), 
                ]).sample(size=(self.nsamples,))
                
        # Compute logQ of amortized posterior samples
        mu, L, Sigma = self.variationalparams(X, indexes)
        q_distribution = RandomFlowSequence(
            source=GaussianSource(D=H, device=device), 
            sequence=[
                LTLinearTransform(LT=L), 
                CopulaTransform(torch.diagonal(Sigma, dim1=-1, dim2=-2)), 
                CollapsedBernoulliLogisticRelaxation(mu=mu, t=self.temperature), 
                ])
        reflow_samples, log_q_s = q_distribution.log_p(relaxed_Kset_samples.permute(0, 2, 1, 3))
        log_q_s = log_q_s.permute(0, 2, 1)
        
        # Compute cross-entropy
        p_s = compute_probabilities(log_p)
        crossH_pq = -torch.sum(p_s * log_q_s) / self.nsamples
        H_p = - torch.sum(p_s * torch.log(p_s))
        KL_pq = crossH_pq - H_p
        
        res = {}
        res["crossH_pq"] = crossH_pq
        res["KL_pq"] = KL_pq

        if self.objective_type == Objective.CROSSENTROPY:
            res["objective"] = crossH_pq
        elif self.objective_type == Objective.KLDIVERGENCE:
            res["objective"] = KL_pq
        else:
            raise NotImplementedError()
        
        if True:  # debug variables
            # Compute data mean and covariance
            res["p_mean"] = sample_mean(Kset, weights=compute_probabilities(log_p), dim=1)
            res["p_covar"] = batch_sample_covar(Kset, weights=compute_probabilities(log_p))
        
            # Compute sample mean and covariance
            q_samples, q_samples_log_p = q_distribution.sample(size=(self.nsamples, 1))
            res["q_samples"] = q_samples
            res["q_samples_log_p"] = q_samples_log_p
            res["q_mean"] = sample_mean(binarize(q_samples))
            res["q_covar"] = batch_sample_covar(binarize(q_samples.permute(1, 0, 2)))
        
            # relaxed_Kset_samples: [nsamples, N:Xdatapoints, K:Ksize, H:Nbits]
            res["relaxed_Kset_samples"] = relaxed_Kset_samples
        
        return res


    def importance_sampling_objective(self, X, Kset, log_f, marginal_p=None, indexes=None):
        """ Returns cross-entropy H(p_K | q(X)) and KL-divergence(p_K || q(X))
            :param X            : [N, D]    N data points
            :param Kset         : [N, K, H] truncated binary posterior sets
            :param log_f        : [N, K]    log-joint probability of (X, Kset) 
            :param marginal_p   : [N, H]    marginal probability of bits
            :param indexes      : [N]       data points indexes
        """
        N, K, H = Kset.shape
        assert Kset.shape[0] == X.shape[0]
        assert Kset.shape[:2] == log_f.shape

        device = X.device

        if marginal_p is None:
            marginal_p = compute_marginal(Kset, log_f)
            marginal_p = torch.clamp(marginal_p, min=0.01, max=0.99)
        
        if self.objective_type == Objective.MEANKLDIVERGENCE:
            # Here we use only mu to compute bits probabilities of q. Ignore correlations
            mu, L, Sigma = self.variationalparams(X, indexes)
            
            q_distribution = RandomFlowSequence(
                source=UniformSource(D=H, device=device), 
                sequence=[
                    ElementwiseLinearTransform(0.9999, 0.00005),  # for stability
                    CollapsedBernoulliLogisticRelaxation(mu=mu, t=self.temperature), 
                    ])
            
            q_s = stable_logistic(mu)
            # Compute cross-entropy        
            crossH_pq = - (marginal_p * torch.log(q_s) + (1-marginal_p) * torch.log(1-q_s)).sum(-1)
            p_s = compute_probabilities(log_f)  # prob. of Kset
            H_marginal_p = - (marginal_p * torch.log(marginal_p) + (1-marginal_p)*torch.log(1-marginal_p)).sum(-1)
            KL_pq = crossH_pq - H_marginal_p
            res = {}
            res["crossH_pq"] = crossH_pq.mean()
            res["KL_pq"] = KL_pq.mean()
            res["objective"] = res["KL_pq"]
        else:
            # Create training samples of relaxed Kset from the auxiliary density (product of marginals)
            relaxed_mu = torch.log((marginal_p)/(1-marginal_p)).unsqueeze(-2)  # inverse logistic of marginal
            relaxed_Kset_samples, relaxed_log_p = RandomFlowSequence(
                source=RelaxedBernoulliSource(marginal=marginal_p, 
                                            Kset=Kset, 
                                            device=device), 
                sequence=[
                    ElementwiseLinearTransform(0.9999, 0.00005),  # for stability
                    CollapsedBernoulliLogisticRelaxation(mu=relaxed_mu, t=self.temperature), 
                    #InverseLogisticTransform(), 
                    #ElementwiseLinearTransform(b=relaxed_mu),
                    #AnnealingTransform(self.temperature), 
                    #LogisticTransform()
                    ]).sample(size=(self.nsamples,))
            
            # Compute logQ of amortized posterior samples
            mu, L, Sigma = self.variationalparams(X, indexes)
            q_distribution = RandomFlowSequence(
                source=GaussianSource(D=H, device=device), 
                sequence=[
                    LTLinearTransform(LT=L), 
                    CopulaTransform(torch.diagonal(Sigma, dim1=-1, dim2=-2)), 
                    CollapsedBernoulliLogisticRelaxation(mu=mu, t=self.temperature), 
                    ElementwiseLinearTransform(1.0001, -0.00005),  # for stability
                    ])
            reflow_samples, log_q_s_samples = q_distribution.log_p(relaxed_Kset_samples.permute(0, 2, 1, 3))
            log_q_s_samples = log_q_s_samples.permute(0, 2, 1)
            
            # Compute cross-entropy        
            r_s = (marginal_p.unsqueeze(-2) * Kset + (1-marginal_p.unsqueeze(-2)) * (1-Kset)).prod(-1)  # prob. of Kset under factorized marginal distribution
            p_s = compute_probabilities(log_f)  # prob. of Kset
            q_s = r_s / self.nsamples * (torch.exp(log_q_s_samples - relaxed_log_p).sum(0))
            crossH_pq = -(p_s * torch.log(q_s)).sum(-1)
            
            H_p = stable_entropy(log_f)
            KL_pq = crossH_pq - H_p
            
            res = {}
            res["crossH_pq"] = crossH_pq.mean()
            res["KL_pq"] = KL_pq.mean()

            if self.objective_type == Objective.CROSSENTROPY:
                res["objective"] = res["crossH_pq"]
            elif self.objective_type == Objective.KLDIVERGENCE:
                res["objective"] = res["KL_pq"]
            else:
                raise NotImplementedError()
            if torch.isnan(res["objective"]):
                assert not torch.isnan(res["objective"])

        if True:  # debug variables
            # Compute data mean and covariance
            res["p_samples"] = torch.stack([k[torch.multinomial(p, num_samples=self.nsamples, replacement=True)] for k, p in zip(Kset, p_s)], dim=1)
            res["p_mean"] = sample_mean(Kset, weights=compute_probabilities(log_f), dim=1)
            res["p_covar"] = batch_sample_covar(Kset, weights=compute_probabilities(log_f))
        
            # Compute sample mean and covariance
            q_samples, q_samples_log_p = q_distribution.sample(size=(self.nsamples, N))
            res["q_samples"] = q_samples
            res["q_samples_log_p"] = q_samples_log_p
            res["q_mean"] = sample_mean(binarize(q_samples))
            res["q_covar"] = batch_sample_covar(binarize(q_samples.permute(1, 0, 2)))
        
            # relaxed_Kset_samples: [nsamples, N:Xdatapoints, K:Ksize, H:Nbits]
            #res["relaxed_Kset_samples"] = relaxed_Kset_samples

            if False:
                print(" relaxed_mu: ", relaxed_mu)
                print(" mu:         ", mu)
                print(" p_mean:     ", res["p_mean"])
                print(" q_mean:     ", res["q_mean"])
        
        return res
    

    def sample_q(self, X, indexes=None, nsamples=1000):
        """ Sample from the fitted density
            :param X            : [N, D]    N data points
            :param indexes      : [N]       data points indexes
            :param nsamples     : int       number of samples
            :returns samples    : [M, N, H]
        """
        mu, L, Sigma = self.variationalparams(X, indexes)
        N, H = mu.shape

        q_distribution = RandomFlowSequence(
            source=GaussianSource(D=H, device=X.device), 
            sequence=[
                LTLinearTransform(LT=L), 
                CopulaTransform(torch.diagonal(Sigma, dim1=-1, dim2=-2)), 
                CollapsedBernoulliLogisticRelaxation(mu=mu, t=self.temperature), 
                ])
        
        q_samples, q_samples_log_p = q_distribution.sample(size=(nsamples, 1))

        return binarize(q_samples)
    