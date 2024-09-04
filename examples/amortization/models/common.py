#import numpy as np
import math
import torch
import numpy as np

EPS = np.finfo("float32").eps  # The difference between 1.0 and the next smallest representable float larger than 1.0


def check_none(x, message):
    if torch.any(torch.isnan(x)) \
            or torch.any(torch.isinf(x)) \
            or torch.any(torch.isneginf(x)):
        print(message)
        assert not torch.any(torch.isnan(x))
        assert not torch.any(torch.isinf(x))
        assert not torch.any(torch.isneginf(x))
        exit()


def zeromean_Gaussian_logprob(x, sigmasqr):
    """ Log-prob of zero mean univariate Gaussian samples
    """
    return -0.5*math.log(2*math.pi) -0.5*torch.log(sigmasqr) -0.5*(1/sigmasqr)*(x**2)


def KL_Gaussian_Gaussian(mu_q, sigmasqr_q, mu_p, sigmasqr_p):
    """ KL(q||p) divergance between two univariate Gaussians
    """
    return 0.5*(torch.log(sigmasqr_p / sigmasqr_q) + (sigmasqr_q + (mu_q-mu_p)**2) / sigmasqr_p - 1)


class RandomSource(torch.nn.Module):

    def sample(self, size):
        raise NotImplementedError()

    def log_p(self, x):
        raise NotImplementedError()


class FlowTransform(torch.nn.Module):
    def forward(self, x, lp):
        raise NotImplementedError()

    def inverse(self, fx):
        raise NotImplementedError()



class RandomFlowSequence(torch.nn.Module):
    def __init__(self, source, sequence) -> None:
        super().__init__()
        self.source = source
        self.sequence = sequence

    def sample(self, size):
        x, lp = self.source.sample(size)
        for transform in self.sequence:
            x_old, lp_old = x, lp
            x, lp = transform(x, lp)
            check_none(x, transform.__class__.__name__ + " sample x")
            check_none(lp, transform.__class__.__name__ + " sample lp")
        return x, lp
        

    def log_p(self, x):
        for transform in reversed([m for m in self.sequence]):
            x_old = x
            x = transform.inverse(x)
            check_none(x, transform.__class__.__name__ + " inverse x")
            
        lp = self.source.log_p(x)
        for transform in self.sequence:
            x_old, lp_old = x, lp
            x, lp = transform(x, lp)
            check_none(x, transform.__class__.__name__ + " log_p x")
            check_none(lp, transform.__class__.__name__ + " log_p lp")
        return x, lp

        
class UniformSource(RandomSource):
    def __init__(self, D, device) -> None:
        super().__init__()
        self.D = D
        self.device = device

    def sample(self, size):
        x = torch.rand(size=size + (self.D,), device=self.device)
        lp = torch.zeros_like(x).sum(-1)  # uniform 1 
        return x, lp

    def log_p(self, fx):
        lp = torch.zeros_like(fx).sum(-1)  # uniform 1 
        return lp
    

class GaussianSource(RandomSource):
    def __init__(self, D, device) -> None:
        super().__init__()
        self.D = D
        self.device = device

    def sample(self, size):
        x = torch.randn(size=size + (self.D,), device=self.device)
        lp = self.log_p(x)
        return x, lp

    def log_p(self, fx):
        lp = -0.5*math.log(2*math.pi) -0.5*(fx**2)
        return lp.sum(-1)
    

class RelaxedBernoulliSource(RandomSource):
    def __init__(self, marginal, Kset, device) -> None:
        super().__init__()
        self.marginal = marginal.unsqueeze(-2)
        self.Kset = Kset
        self.device = device

    def sample(self, size):
        N, K, H = self.Kset.shape
        uniformsamples = torch.rand(size=size + (N, K, H), device=self.device)
        x = uniformsamples * (1-self.marginal) * (1-self.Kset) + \
            (uniformsamples * self.marginal + (1-self.marginal)) * self.Kset
        #x = 0*x + self.marginal  # for debugging
        lp = torch.zeros_like(x).sum(-1)  # uniform 1 
        return x, lp

    def log_p(self, fx):
        lp = torch.zeros_like(fx).sum(-1)  # uniform 1
        return lp


    
class LTLinearTransform(FlowTransform):
    def __init__(self, LT, mean=None) -> None:
        super().__init__()
        self.LT = LT  # lower triangular matrix
        self.mean = mean  # additive mean vector

    def forward(self, x, lp):
        assert self.LT is not None
        logdet = torch.log(torch.abs(torch.diagonal(self.LT, dim1=-1, dim2=-2))).sum(-1)
        fx = torch.matmul(self.LT, x.unsqueeze(-1)).squeeze(-1) 
        if self.mean is not None:
            fx += self.mean
        return fx, lp - logdet

    def inverse(self, fx):
        if self.mean is not None:
            fx -= self.mean
        return torch.matmul(torch.inverse(self.LT), fx.unsqueeze(-1)).squeeze(-1)
        

class CopulaTransform(FlowTransform):
    def __init__(self, diagcovars=None) -> None:
        super().__init__()
        self.diagcovars = diagcovars  # marginal covariances of the dimensions

    def forward(self, x, lp):
        assert self.diagcovars is not None
        logdet = (-0.5*math.log(2*math.pi) -0.5*torch.log(self.diagcovars) -0.5*(1/self.diagcovars)*(x**2)).sum(-1)
        fx = 0.5 * (1 + torch.erf(x/torch.sqrt(2*self.diagcovars))) 
        check_none(lp, "Copula.forward lp")
        check_none(x, "Copula.forward x")
        check_none(logdet, "Copula.forward logdet")
        check_none(fx, "Copula.forward fx")
        check_none(lp - logdet, "Copula.forward lp - logdet")
        fx = torch.clamp(fx, EPS, 1-EPS)
        return fx, lp - logdet

    def inverse(self, fx):
        fx = torch.clamp(fx, EPS, 1-EPS)
        x = torch.erfinv(2*fx-1) * torch.sqrt(2*self.diagcovars)
        check_none(x, "Copula.inverse x")        
        return x


class ElementwiseLinearTransform(FlowTransform):
    def __init__(self, a=1, b=0) -> None:
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, x, lp):
        if isinstance(self.a, torch.Tensor):
            logdet = torch.log(self.a).sum(-1)
        else:
            logdet = math.log(self.a) * x.shape[-1]
        fx = self.a*x + self.b
        return fx, lp - logdet

    def inverse(self, fx):
        return (fx-self.b) * (1 / self.a)


class ScaleTransform(FlowTransform):
    def __init__(self, scale=1) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x, lp):
        fx = self.scale * x
        if isinstance(self.scale, torch.Tensor):
            logdet = torch.log(self.scale).sum(-1)
        else:
            logdet = x.shape[-1] * math.log(self.scale)
        return fx, lp - logdet

    def inverse(self, fx):
        return fx / self.scale


class AnnealingTransform(ScaleTransform):
    def __init__(self, t=1) -> None:
        super().__init__(scale=1/t)

    def set_temperature(self, temperature):
        self.scale = 1 / temperature

    def get_temperature(self):
        return 1 / self.scale


def stable_logistic(x):
    # This is much more numerically stable than simple
    # return 1 / (1 + torch.exp(-x))
    fx = torch.where(x>0, 1 / (1 + torch.exp(-x)),
                          torch.exp(x) / (1 + torch.exp(x)))
    return fx


def stable_inv_logistic(x):
    x = torch.clamp(x, EPS, 1-EPS)
    fx = torch.log(x/(1-x))
    return fx


class LogisticTransform(FlowTransform):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x, lp):
        fx = stable_logistic(x)
        logdet = (-x - 2*torch.log(torch.exp(-x) + 1)).sum(-1)  
        # logdet = (-x + 2*torch.log(fx)).sum(-1)
        return fx, lp - logdet
    
    def inverse(self, fx):
        return stable_inv_logistic(fx)


class InverseLogisticTransform(FlowTransform):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x, lp):
        fx = stable_inv_logistic(x)
        logdet = -torch.log(torch.clamp(x-x**2, EPS, 1-EPS)).sum(-1)
        return fx, lp - logdet
    
    def inverse(self, fx):
        return stable_logistic(fx)


class CollapsedBernoulliLogisticRelaxation(FlowTransform):
    """ Collapsed (combined and simplified) transformation:
        inverse-logistic -> shift my mean -> temperature -> logistic
    """
    def __init__(self, mu, t) -> None:
        super().__init__()
        self.mu = mu  # mean parameter
        self.t = t  # temperature
        self.invt = 1/t  # inverse temperature

    def forward(self, x, lp):
        x = torch.clamp(x, EPS, 1-EPS)
        g = (1-x)/x * torch.exp(-self.mu)
        fx = 1 / (g**self.invt + 1)
        fx = torch.clamp(fx, EPS, 1-EPS)
        logdet = (torch.log(self.invt * g**(self.invt-1)) - self.mu - 2*torch.log(x) + 2*torch.log(fx)).sum(-1)
        #logdet = (torch.log(self.invt) + (self.invt-1) * torch.log(g) - self.mu - 2*torch.log(x) + 2*torch.log(fx)).sum(-1)
        return fx, lp - logdet
    
    def inverse(self, fx):
        fx = torch.clamp(fx, EPS, 1-EPS)
        x = 1 / ((1 / fx - 1)**self.t * torch.exp(self.mu) + 1)
        x = torch.clamp(x, EPS, 1-EPS)
        return x


if __name__ == "__main__":
    x, lp = GaussianSource(D=2, device="cpu").sample(size=(1000,))
    x = 2.0 * x
    x1, lp1 = LogisticTransform().forward(x, lp)
    x2, lp2 = InverseLogisticTransform().forward(x1, lp1)
    err = torch.abs(lp - lp2).max()
    print("log-prob error: {}".format(err))
    assert err < 1e-3
    

if __name__ == "__main__":
    x, lp = GaussianSource(D=2, device="cpu").sample(size=(1000,))
    
    x1, lp1 = LogisticTransform().forward(x, lp)
    x1, lp1 = InverseLogisticTransform().forward(x1, lp1)
    x1, lp1 = ElementwiseLinearTransform(b=1).forward(x1, lp1)
    x1, lp1 = AnnealingTransform(2.0).forward(x1, lp1)
    x1, lp1 = LogisticTransform().forward(x1, lp1)
    
    x2, lp2 = LogisticTransform().forward(x, lp)
    x2, lp2 = CollapsedBernoulliLogisticRelaxation(mu=torch.Tensor([1.0]), t=torch.Tensor([2.0])).forward(x2, lp2)

    err = torch.abs(x1 - x2).max()
    print("x error: {}".format(err))
    assert err < 1e-3

    err = torch.abs(lp1 - lp2).max()
    print("log-prob error: {}".format(err))
    assert err < 1e-3
    
if __name__ == "__main__":
    x = torch.tensor([-EPS, 0, EPS, 0.5, 1-EPS, 1, 1+EPS], dtype=torch.float32)
    print(x)
    print(torch.log(x))
    inv_logistic_x = stable_inv_logistic(x)
    print(inv_logistic_x)
    print(1e30*inv_logistic_x)
    x_star = stable_logistic(1e30*inv_logistic_x)
    print(x_star)