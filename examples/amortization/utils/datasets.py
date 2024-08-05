import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset



class KSetPosteriorDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.X = None  # [N, D]
        self.Kset = None  # [N, K, H]
        self.logPs = None  # [N, K]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Kset[idx], self.logPs[idx], idx

    def to(self, device):
        self.X = self.X.to(device)
        self.Kset = self.Kset.to(device)
        self.logPs = self.logPs.to(device)


class ToyDatasetH1(KSetPosteriorDataset):
    def __init__(self) -> None:
        super().__init__()
        self.X = torch.Tensor([[1, 2, 3, 4], 
                               [5, 6, 7, 8],
                               [8, 7, 6, 5],
                               ])
        self.Kset = torch.Tensor([[[0], [1]],
                                  [[0], [1]],
                                  [[0], [1]],
                                  ])
        self.logPs = torch.Tensor([[0.1, 0.9],
                                   [0.5, 0.5],
                                   [0.9, 0.1],
                                   ]).log()


class ToyDatasetH2(KSetPosteriorDataset):
    def __init__(self) -> None:
        super().__init__()
        self.X = torch.Tensor([[1, 2, 3, 4], 
                               [5, 6, 7, 8],
                               [8, 7, 6, 5],])
        self.Kset = torch.Tensor([[[0, 0], 
                                   [1, 0],
                                   [0, 1],
                                   [1, 1]],
                                  [[0, 0], 
                                   [1, 0],
                                   [0, 1],
                                   [1, 1]],
                                  [[0, 0], 
                                   [1, 0],
                                   [0, 1],
                                   [1, 1]],])
        self.logPs = torch.Tensor([[0.001, 0.5, 0.5, 0.001],
                                   [0.5, 0.001, 0.001, 0.5],
                                   [0.1, 0.3, 0.3, 0.3],
                                   ]).log()
        

class ToyDatasetH2Minimal(KSetPosteriorDataset):
    def __init__(self) -> None:
        super().__init__()
        self.X = torch.Tensor([[1, 2, 3, 4]])
        self.Kset = torch.Tensor([[[0, 0], 
                                   [1, 0],
                                   [0, 1],
                                   [1, 1]],])
        self.logPs = torch.Tensor([[0.001, 0.1, 0.8, 0.001]]).log()


class LargeCorrelatedDataset(KSetPosteriorDataset):
    def __init__(self, Hcorr=10, Huncorr=10, K=100) -> None:
        super().__init__()
        N = 1
        D = 100

        # Construct a covariance matrix with linear kernel
        x = np.linspace(-1, 1, Hcorr)
        Kxx = x[:, np.newaxis] * x[np.newaxis, :] + 0.03 * np.eye(Hcorr)
        gaussian_samples = np.random.multivariate_normal(mean=np.zeros_like(x), cov=Kxx, size=K)
        corr_samples = (gaussian_samples > 0)*1
        
        threshold = np.random.random_sample(Huncorr)
        uncorr_samples = (np.random.random_sample(size=(K, Huncorr)) > threshold)*1
        binary_samples = np.concatenate([corr_samples, uncorr_samples], axis=-1)
        
        self.X = torch.Tensor(np.random.random_sample(size=(N, D)))
        self.Kset = torch.Tensor(binary_samples[np.newaxis, ...])
        self.logPs = torch.Tensor(np.ones(shape=(N, K)))


class TVODataset(KSetPosteriorDataset):
    def __init__(self, Xpath, Ksetpath, start=0, maxN=None) -> None:
        super().__init__()
        end = None if maxN is None else start+maxN

        with h5py.File(Xpath, "r") as f:
            self.X = torch.tensor(np.array(f["data"])[start:end], dtype=torch.get_default_dtype())

        with h5py.File(Ksetpath, "r") as f:
            self.Kset = torch.tensor(np.array(f["train_states"])[start:end])
            self.logPs = torch.tensor(np.array(f["train_lpj"])[start:end], dtype=torch.get_default_dtype())


class XDataset(Dataset):
    def __init__(self, Xpath, start=0, maxN=None) -> None:
        super().__init__()
        end = None if maxN is None else start+maxN

        with h5py.File(Xpath, "r") as f:
            self.X = torch.tensor(np.array(f["data"])[start:end], dtype=torch.get_default_dtype())

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], idx

    def to(self, device):
        self.X = self.X.to(device)
        

def compute_probabilities(log_p_joint):
    """ Returns vectors probabilities (normalized)
        :param log_f        : [N, K]    log-joint probability of (X, Kset) 
    """
    Ps = log_p_joint.exp() / (log_p_joint.exp().sum(-1).unsqueeze(-1))
    return Ps

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
    #return torch.cov(x.T, aweights=weights, correction=0)
    if weights is not None:
        x_mean = sample_mean(x, weights)
        x_zeromean = x - x_mean
        x_weighted = torch.sqrt(weights).unsqueeze(-1) * x_zeromean
        x_cov = torch.mm(x_weighted.T, x_weighted)
        return x_cov
    return torch.cov(x.T, aweights=weights, correction=0)


if __name__ == "__main__":
    path = "../../../../relaxed-bernoulli-datasets/House12/sigma25"
    dataset = TVODataset(path)
    print(dataset.Kset.shape)

    import matplotlib.pyplot as plt
    for Kset, logPs in zip(dataset.Kset, dataset.logPs):
        #plt.imshow(Kset)
        #plt.show()

        cov = sample_covar(Kset, compute_probabilities(logPs))
        tcov = torch.cov(Kset.T, aweights=compute_probabilities(logPs), correction=0)

        plt.imshow(cov, vmin=-1, vmax=1, cmap="seismic")
        plt.show()