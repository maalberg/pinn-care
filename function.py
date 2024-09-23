from abc import abstractmethod
from abc import ABCMeta as interface

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------*/
# - function

class function(metaclass=interface):
    @abstractmethod
    def __call__(self, domain: tf.Tensor) -> tf.Tensor:
        """
        Sample this function on given ``domain``.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters(self) -> tf.Tensor:
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
    def __init__(self, arch: list[int], acts: list[str]) -> None:

        # construct a sequential model to wrap network layers
        self._model = tf.keras.Sequential(name='fcnn')

        # specify input data shape;
        # this also makes the model build its layers (and thus weights) automatically
        self._model.add(tf.keras.Input(shape=(arch[0],)))

        for neurons_n, act in zip(arch[1:-1], acts[:-1]):
            self._model.add(tf.keras.layers.Dense(neurons_n, activation=act))

        self._model.add(tf.keras.layers.Dense(arch[-1], activation=acts[-1], use_bias=False))

    def __call__(self, domain: tf.Tensor) -> tf.Tensor:
        return self._model(domain)

    @property
    def parameters(self) -> tf.Tensor:
        return self._model.trainable_variables
