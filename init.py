import torch
from torch import optim, nn
import copy
from tqdm import tqdm
from sacred.observers import FileStorageObserver

class ZeroOutput(nn.Module):
    """Zero the output of a model by subtracting out a copy of it."""

    def __init__(self, model):
        super().__init__()
        self.init_model = [copy.deepcopy(model).eval()]

        self.model = model

    def forward(self, inp):
        return self.model(inp) - self.init_model[0](inp)

ACTS = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh
}

def simple_net(width,
               bias=True,
               zero_output=True,
               hidden_layers=1,
               act='relu',
               **kwargs):
    """A simple 1d input to 1d output deep ReLU network.

    Parameters
    ----------
    bias : bool
        Whether to include biases.
    zero_output : bool
        Whether to zero the output of the model.
    """
    a = ACTS[act]
    model = nn.Sequential(nn.Linear(1, width, bias=bias),
                          a(),
                          *[layer
                            for _ in range(hidden_layers-1)
                            for layer in [nn.Linear(width, width, bias=bias), a()]
                            ],
                          nn.Linear(width, 1, bias=bias))

    if zero_output:
        model = ZeroOutput(model)
    return model

