import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_layers,
        activations
    ):
        super().__init__()


        activation_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh
        }

        layers = []
        in_dim = input_dim

        for h, act in zip(hidden_layers, activations):
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation_map[act]())

            in_dim = h

        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
