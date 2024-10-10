import numpy as np
import torch


# ---------------------------------------------------------------------------*/
# - pytorch-based fully-connected neural network

class fcnn(torch.nn.Module):
    def __init__(self, features: list[int], activations) -> None:
        super().__init__()

        # use python's list comprehension to construct a fully-connected neural network with a one-liner
        self._model = torch.nn.Sequential(*[
            torch.nn.Sequential(*[
                torch.nn.Linear(i, o, bias=a[1]), a[0]()]) for i, o, a in zip(features[:-1], features[1:], activations)])

    def forward(self, domain: torch.Tensor) -> torch.Tensor:
        return self._model(domain)
