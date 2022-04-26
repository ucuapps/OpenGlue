import numpy as np
import torch
import torch.nn as nn


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


# https://github.com/vsitzmann/siren
class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


class FeedForwardNetSiren(nn.Sequential):
    def __init__(self, *args):
        layers = []
        for i in range(1, len(args) - 1):
            layers.extend((
                nn.Conv1d(args[i - 1], args[i], kernel_size=1),
                Sine()
                # nn.BatchNorm1d(args[i])
            ))
        layers.append(nn.Conv1d(args[-2], args[-1], kernel_size=1))
        for l in layers:
            l.apply(sine_init)
        layers[0].apply(first_layer_sine_init)
        super(FeedForwardNetSiren, self).__init__(*layers)


class FeedForwardNet(nn.Sequential):
    def __init__(self, *args):
        layers = []
        for i in range(1, len(args) - 1):
            layers.extend((
                nn.Conv1d(args[i - 1], args[i], kernel_size=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(args[i])
            ))
        layers.append(nn.Conv1d(args[-2], args[-1], kernel_size=1))
        super(FeedForwardNet, self).__init__(*layers)
