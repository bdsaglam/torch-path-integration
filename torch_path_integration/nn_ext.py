import torch.nn as nn
import torch.nn.functional as F


def MLP(sizes, activation=nn.ReLU, activate_final=True, bias=True):
    n = len(sizes)
    assert n >= 2, "There must be at least two sizes"

    layers = []
    for j in range(n - 1):
        layers.append(nn.Linear(sizes[j], sizes[j + 1], bias=bias))
        layers.append(activation())

    if not activate_final:
        layers.pop()

    return nn.Sequential(*layers)


def relu1(x):
    return F.relu6(x * 6.) / 6.
