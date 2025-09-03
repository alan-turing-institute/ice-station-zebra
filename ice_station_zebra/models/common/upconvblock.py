from torch import Tensor, nn
from typing import Optional

from .activations import ACTIVATION_FROM_NAME


class UpconvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "ReLU",
        normalization: Optional[str] = "batch",  # None, "batch", or "group"
        num_groups: Optional[int] = None,  # Required for GroupNorm
    ) -> None:
        """
        Initialise a flexible UpconvBlock.
        
        Args:
            in_channels: Input channel size
            out_channels: Output channel size
            activation: Activation function name
            normalization: Type of normalization (None, "batch", or "group")
            num_groups: Number of groups for GroupNorm (required if normalization="group")
        """
        super().__init__()
        
        if normalization == "group" and num_groups is None:
            raise ValueError("num_groups must be specified when using GroupNorm")
        
        activation_layer = ACTIVATION_FROM_NAME[activation]
        
        layers = []
        
        # Upsampling layer
        layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
        
        # Convolution
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=2, padding="same"))
        
        # Optional normalization
        if normalization == "batch":
            layers.append(nn.BatchNorm2d(num_features=out_channels))
        elif normalization == "group":
            layers.append(nn.GroupNorm(num_groups, out_channels))
        
        # Activation
        layers.append(activation_layer(inplace=True))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
