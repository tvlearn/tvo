import math
import torch
import torch.nn as nn
from torch.nn import Module, Parameter



def isbatchofsquarematrices(A):
    return A.dim() >= 3 and A.shape[-1] == A.shape[-2]


def btrace(a):
    return torch.diagonal(a, dim1=-2, dim2=-1).sum(-1) 


def bdiag(a):
    return torch.diagonal(a, dim1=-2, dim2=-1)


def bprod(a):
    return torch.diagonal(a, dim1=-2, dim2=-1).prod(-1) 


def multivar_normal_entropy(L):
    """ Entropy of multivariate Gaussian with covariance matrix 
        given by its lower-triangular (Cholesky) decomposition.
    """
    # L is lower triangular matrix from the Cholesky decomposition
    #Sigma_logdet = 2 * torch.log(torch.abs(bprod(L)))
    if isbatchofsquarematrices(L):
        L = bdiag(L)
    Sigma_logdet = 2 * torch.log(torch.abs(L)).sum(-1)  # more stable
    entropy = 0.5 * Sigma_logdet + 0.5 * L.shape[-1] * math.log(2*math.pi*math.e)
    return entropy


def unit_Laplace_logprob(x):
    """ Log-prob of Laplace samples with b=1, mu=0
    """
    return -torch.abs(x) - math.log(2.0)


def unit_Gaussian_logprob(x):
    """ Log-prob of Gaussian samples with mu=0, sigmasqr=1
    """
    return -0.5*math.log(2*math.pi) -0.5*(x**2)


def zeromean_Gaussian_logprob(x, sigmasqr):
    """ Log-prob of zero mean univariate Gaussian samples
    """
    return -0.5*math.log(2*math.pi) -0.5*torch.log(sigmasqr) -0.5*(1/sigmasqr)*(x**2)


def cholesky_jitter(A, maxattempts=10):
    """ Iteratively try adding larger diagonal to perform Cholesky decomposition
        Maybe use https://github.com/cornellius-gp/linear_operator/blob/main/linear_operator/utils/cholesky.py
    """
    try:
        return torch.linalg.cholesky(A)
    except Exception as err:
        if err.__class__.__name__ != "_LinAlgError":
            raise
    
    for i in range(maxattempts):
        jitter = 1e-9 * 10**i
        print("Cholesky: adding diagonal {:.2e}".format(jitter))
        try:
            return torch.linalg.cholesky(A + jitter*torch.eye(A.shape[-1], device=A.device))
        except Exception as err:
            if err.__class__.__name__ != "_LinAlgError":
                raise
            

class VariationalParams(Module):
    def __init__(self, N, D, H) -> None:
        super().__init__()
        self.N = N  # number of data points
        self.D = D  # observed space dimensionality
        self.H = H  # latent space dimensionality


    def forward(self, X, indexes):
        raise NotImplementedError()
    


class FullCovarGaussianVariationalParams(VariationalParams):
    def __init__(self, N, D, H) -> None:
        super().__init__(N, D, H)
        # posterior mean
        self.mu = Parameter(torch.zeros(size=(self.N, self.H)))
        # posterior covariance lower triangular parameters
        self.Sigma_param = Parameter(torch.eye(self.H).reshape(1, self.H, self.H).repeat(self.N, 1, 1))
        

    def forward(self, X, indexes=None):
        if indexes is None:
            L = torch.tril(self.Sigma_param)
            Sigma = torch.bmm(L, torch.transpose(L, -1, -2))
            return self.mu, L, Sigma
        else:
            L = torch.tril(self.Sigma_param[indexes])
            Sigma = torch.bmm(L, torch.transpose(L, -1, -2))
            return self.mu[indexes], L, Sigma


    def set(self, indexesto, other, indexesfrom):
        with torch.no_grad():
            self.mu[indexesto] = other.mu[indexesfrom]
            self.Sigma_param[indexesto] = other.Sigma_param[indexesfrom]
        


class DiagCovarGaussianVariationalParams(VariationalParams):
    def __init__(self, N, D, H) -> None:
        super().__init__(N, D, H)
        # posterior mean
        self.mu = Parameter(torch.zeros(size=(self.N, self.H)))
        # posterior covariance lower triangular parameters
        self.Sigma_param = Parameter(torch.ones(size=(self.N, self.H)))
        

    def forward(self, X, indexes=None):
        if indexes is None:
            L = self.Sigma_param
            Sigma = L**2
            return self.mu, L, Sigma
        else:
            L = self.Sigma_param[indexes]
            Sigma = L**2
            return self.mu[indexes], L, Sigma


    def set(self, indexesto, other, indexesfrom):
        with torch.no_grad():
            self.mu[indexesto] = other.mu[indexesfrom]
            self.Sigma_param[indexesto] = other.Sigma_param[indexesfrom]
    


class AmortizedDiagCovarGaussianVariationalParams(VariationalParams):
    def __init__(self, N, D, H, rank=5) -> None:
        super().__init__(N, D, H)
        self.rank = rank
        nnSize = (self.D, 2*self.D, 4*self.D, self.H)
        
        # Shared NN
        self.nnShared = nn.Sequential(
                nn.Linear(self.D, nnSize[0]), 
                nn.ReLU(),
                )

        # NN for means
        self.nnMeans = nn.Sequential(
                nn.Linear(nnSize[0], nnSize[1]), 
                nn.ReLU(),
                nn.Linear(nnSize[1], nnSize[2]), 
                nn.ReLU(),
                nn.Linear(nnSize[2], nnSize[3]), 
                )
        
        # NN for diagonal covars
        self.nnDiagCovars = nn.Sequential(
                nn.Linear(nnSize[0], nnSize[1]), 
                nn.ReLU(),
                nn.Linear(nnSize[1], nnSize[2]), 
                nn.ReLU(),
                nn.Linear(nnSize[2], nnSize[3]), 
                #nn.Softplus(),
                )
        
        

    def forward(self, X, indexes):
        shared = self.nnShared(X)
        mu = self.nnMeans(shared)
        L = self.nnDiagCovars(shared) + 0.001
        Sigma = L**2
        return mu, L, Sigma
    


class AmortizedGaussianVariationalParams(VariationalParams):
    def __init__(self, N, D, H, rank=5) -> None:
        super().__init__(N, D, H)
        self.rank = rank
        nnSize = (self.D, 10*self.D, 10*self.D, self.H)
        
        # NN for means
        self.nnMeans = nn.Sequential(
                nn.Linear(nnSize[0], nnSize[1]), 
                nn.ReLU6(),
                nn.Linear(nnSize[1], nnSize[2]), 
                nn.ReLU6(),
                nn.Linear(nnSize[2], nnSize[3]), 
                )
        
        # NN for diagonal covars
        self.nnDiagCovars = nn.Sequential(
                nn.Linear(nnSize[0], nnSize[1]), 
                nn.ReLU6(),
                nn.Linear(nnSize[1], nnSize[2]), 
                nn.ReLU6(),
                nn.Linear(nnSize[2], nnSize[3]), 
                nn.Softplus(),
                )
        
        # NN for low-rank covar
        self.nnLowRankCovar = nn.Sequential(
                nn.Linear(nnSize[0], nnSize[1]), 
                nn.ReLU6(),
                nn.Linear(nnSize[1], nnSize[2]), 
                nn.ReLU6(),
                nn.Linear(nnSize[2], rank*nnSize[3]), 
                )
        

    def forward(self, X, indexes):
        mu = self.nnMeans(X)
        V = 1e-6 * self.nnLowRankCovar(X).reshape((X.shape[0], self.H, -1))
        Sigma = torch.bmm(V, V.transpose(-1, -2)) + torch.diag_embed(self.nnDiagCovars(X))
        L = cholesky_jitter(Sigma)
        return mu, L, Sigma

