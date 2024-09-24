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
        """
        The architecture of this fully-connected neural network is defined by
        ``arch``. This parameter is a list of integers, where the integers
        specify the number of units in the network layers. For example,
        [784 30 10] defines a typical mnist network, where the number
        of input image pixels is 784, the single hidden layer has
        30 neurons, and there are 10 digit probabilities to
        output. The neuron activations are defined by
        ``acts`` - a list of strings. For the mnist
        example above this list could be ['sigmoid', 'sigmoid']. Note that the
        first layer, i.e. the input, has no activation, so the size of
        ``acts`` is the size of ``arch`` minus one. Provided this
        network is designed for regression, and the output
        should have no activation, then the last string
        must be 'linear'.
        """

        # construct a sequential model to wrap fully-connected network layers
        self._model = tf.keras.Sequential(name='fcnn')

        # specify input layer;
        # in tensorflow this is done by specifying input data shape,
        # which also makes the model build its layers (and thus weights) automatically
        self._model.add(tf.keras.Input(shape=(arch[0],)))

        # add hidden layers to the model
        for neurons_n, neurons_act in zip(arch[1:-1], acts[:-1]):
            self._model.add(tf.keras.layers.Dense(neurons_n, activation=neurons_act))

        # add output layer;
        # note that the output layer has no bias
        self._model.add(tf.keras.layers.Dense(arch[-1], activation=acts[-1], use_bias=False))

    def __call__(self, domain: tf.Tensor) -> tf.Tensor:
        return self._model(domain)

    def print(self) -> None:
        """
        Print the layer structure of this fully-connected neural network.
        """
        print(self._model.summary())

    def save(self, name: str) -> None:
        """
        Save this neural network to keras-formatted file, named ``name``.keras .
        """
        self._model.save(name + '.keras')

    @property
    def parameters(self) -> tf.Tensor:
        return self._model.trainable_variables
