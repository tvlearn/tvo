import torch as to
from torch import Tensor
from tvo.variational.TVOVariationalStates import TVOVariationalStates
from tvo.utils.model_protocols import Optimized, Trainable
from tvo.variational.evo import EVOVariationalStates
from tvo.variational.neural import NeuralVariationalStates


class NeuralEvoVariationalStates(TVOVariationalStates):
    def __init__(
        self,
        N: int,
        H: int,
        S: int,
        precision: to.dtype,
        parent_selection: str,
        mutation: str,
        n_parents: int,
        n_generations: int,
        K_init: Tensor = None,
        n_children: int = None,
        crossover: bool = False,
        bitflip_frequency: float = None,
        K_init_file: str = None,
        update_decoder: bool = True,
        encoder: str = "MLP",
        sampling: str = "Gumbel",
        bitflipping: str = "sparseflip",
        n_samples: int = 1,
        lr: float = 1e-3,
        training=True,
        k_updating=True,
        scheduler: list = None,
        **kwargs,
    ):
        """
        Sampling method that combines the neural and evo approaches
        todo: comment further
        """
        # init evo
        self.evo = EVOVariationalStates(
            N,
            H,
            S,
            precision,
            parent_selection,
            mutation,
            n_parents,
            n_generations,
            n_children=n_children,
            crossover=crossover,
            bitflip_frequency=bitflip_frequency,
            K_init_file=K_init_file,
        )

        self.neural = NeuralVariationalStates(
            N,
            H,
            S,
            precision,
            K_init=K_init,
            update_decoder=update_decoder,
            encoder=encoder,
            sampling=sampling,
            bitflipping=bitflipping,
            n_samples=n_samples,
            lr=lr,
            training=training,
            k_updating=k_updating,
            **kwargs,
        )

        self.scheduler = scheduler

    def update(self, idx: Tensor, batch: Tensor, model: Trainable) -> int:
        """Generate new variational states, update K and lpj with best samples and their lpj.

        :param idx: data point indices of batch w.r.t. K
        :param batch: batch of data points
        :param model: the model being used
        :returns: average number of variational state substitutions per datapoint performed
        """
        cutoff = len(self.scheduler)
        idx1 = idx[0:cutoff]
        batch1 = batch[0:cutoff]
        idx2 = idx[cutoff:-1]
        batch2 = batch[cutoff:-1]
        self.evo.update(idx1, batch1, model)
        self.neural.update(idx2, batch2, model)
