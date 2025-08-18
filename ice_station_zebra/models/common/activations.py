import torch.nn as nn

ACTIVATIONS = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "ELU": nn.ELU,
    "GELU": nn.GELU,
    "SiLU": nn.SiLU,
    "Sigmoid": nn.Sigmoid,
    "Tanh": nn.Tanh,
}

def get_activation(name: str) -> nn.Module:
    if name not in ACTIVATIONS:
        raise ValueError(f"Unsupported activation: {name}")
    return ACTIVATIONS[name]()
