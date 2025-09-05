from torch import Tensor, nn

from .activations import ACTIVATION_FROM_NAME


class ConvNormAct(nn.Module):
    """Mini block: Conv2d -> Norm -> Activation, optional Dropout"""
    def __init__( 
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        norm_type: str = "groupnorm",
        num_groups: int | None = None,
        activation: str = "ReLU",
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        norm_type = norm_type.lower()
        if norm_type == "groupnorm":
            if num_groups is None:
                raise ValueError("num_groups must be specified for GroupNorm")
            norm_layer = nn.GroupNorm(num_groups, out_channels)
        elif norm_type == "batchnorm":
            norm_layer = nn.BatchNorm2d(out_channels)
        elif norm_type == "none":
            norm_layer = nn.Identity()
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}. Choose 'groupnorm', 'batchnorm', or 'none'")
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding="same"),
            norm_layer,
            ACTIVATION_FROM_NAME[activation](inplace=True),
        ]
        if dropout_rate > 0.0:
            layers.append(nn.Dropout2d(dropout_rate))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

class CommonConvBlock(nn.Module):
    """Full block: two ConvNormAct stacked, optionally add a final layer"""
    def __init__(  # Fixed: double underscores
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        norm_type: str = "groupnorm",
        activation: str = "SiLU",
        final: bool = False,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.final = final
        norm_type = norm_type.lower()
        
        # Determine num_groups only if using GroupNorm
        num_groups = self._get_num_groups(out_channels) if norm_type == "groupnorm" else None
        
        # Create stacked ConvNormAct blocks
        self.block1 = ConvNormAct(
            in_channels, out_channels, kernel_size, norm_type, num_groups, activation, dropout_rate
        )
        self.block2 = ConvNormAct(
            out_channels, out_channels, kernel_size, norm_type, num_groups, activation, dropout_rate
        )
        
        self.final_layer = ConvNormAct(
            out_channels, out_channels, kernel_size, norm_type, num_groups, activation, dropout_rate
        ) if final else None
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        if self.final_layer is not None:
            x = self.final_layer(x)
        return x
    
    def _get_num_groups(self, channels: int) -> int:  # Fixed: instance method (not static)
        """
        Determines the maximum number of groups that divide `channels` for GroupNorm.
        Args:
            channels (int): Number of feature channels.
        Returns:
            int: Optimal number of groups.
        """
        num_groups = 8  # Start with preferred group count
        while num_groups > 1:
            if channels % num_groups == 0:
                return num_groups
            num_groups -= 1
        return 1  # Fallback to GroupNorm(1,...), equivalent to LayerNorm

# # Usage examples:

# # GroupNorm without dropout, no final layer
# block_gn = CommonConvBlock(
#     in_channels=64,
#     out_channels=128,
#     norm_type="groupnorm",
#     dropout_rate=0.0,
#     final=False
# )

# # BatchNorm without dropout, no final layer
# block_bn = CommonConvBlock(
#     in_channels=64,
#     out_channels=128,
#     norm_type="batchnorm",
#     dropout_rate=0.0,
#     final=False
# )

# # No normalization, no dropout
# block_none = CommonConvBlock(
#     in_channels=64,
#     out_channels=128,
#     norm_type="none",
#     dropout_rate=0.0,
#     final=False
# )

# # GroupNorm with extra final layer
# block_final = CommonConvBlock(
#     in_channels=64,
#     out_channels=128,
#     norm_type="groupnorm",
#     dropout_rate=0.0,
#     final=True
# )

# # GroupNorm with dropout (10%) but no final layer
# block_dropout = CommonConvBlock(
#     in_channels=64,
#     out_channels=128,
#     norm_type="groupnorm",
#     dropout_rate=0.1,
#     final=False
# )

# # GroupNorm with both final layer and dropout
# block_final_dropout = CommonConvBlock(
#     in_channels=64,
#     out_channels=128,
#     norm_type="groupnorm",
#     dropout_rate=0.1,
#     final=True
# )
