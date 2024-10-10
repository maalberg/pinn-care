from abc import abstractmethod
from abc import ABCMeta as interface

import numpy as np
import torch

import utilities as utils


# ---------------------------------------------------------------------------*/
# - function

class function(metaclass=interface):
    @abstractmethod
    def __call__(self, domain: torch.Tensor) -> torch.Tensor:
        """
        Sample this function on given ``domain``.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters(self) -> torch.Tensor:
        """
        Parameters of this function.
        """
        raise NotImplementedError

    @property
    def dims_i_n(self) -> int: return self.parameters.shape[1]

    @property
    def dims_o_n(self) -> int: return self.parameters.shape[0]


# ---------------------------------------------------------------------------*/
# - neural network

class neuralnetwork(function):
    @staticmethod
    def load(name: str):
        """
        A static method to load a neural network from a file called ``name``.
        The method augments the ``name`` with proper folder
        and file extension.
        """
        self = neuralnetwork()
        self._fcnn = torch.load('data/' + name + '.pt', weights_only=False)
        return self

    @staticmethod
    def build(features: list[int], activations):
        """
        A static method to build a neural network with given ``features`` and ``activations``.
        In principle, ``features`` describe the number of neurons in each network layer,
        e.g. [1, 16, 16, 1] describes a single-input, single-output network with
        two hidden layers, 16 neurons each.
        """
        self = neuralnetwork()
        self._fcnn = utils.fcnn(features, activations)
        return self

    def __call__(self, domain: torch.Tensor) -> torch.Tensor:
        return self._fcnn(domain)

    @property
    def parameters(self) -> list[torch.Tensor]:
        return list(self._fcnn.parameters())

    @property
    def dims_i_n(self) -> int: return self.parameters[0].shape[1]

    @property
    def dims_o_n(self) -> int: return self.parameters[-1].shape[0]

    def print(self) -> None:
        print(self._fcnn)

    def save(self, name: str) -> None:
        torch.save(self._fcnn, 'data/' + name + '.pt')

