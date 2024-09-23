import torch.nn as nn
from torch import Tensor

class ResidualConnection(nn.Module):
    def forward(self, x: Tensor, sublayer: nn.Module) -> Tensor:
        return (x+sublayer(x))