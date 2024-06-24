import torch
from torch import nn
import torch.nn.functional as F
import math
import warnings
from kan_linear import KANLinear

class KANConv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int or tuple = 3,  # type: ignore
        stride: int or tuple = 1,  # type: ignore
        padding: int or str = 0,  # type: ignore
        dilation: int or tuple = 1,  # type: ignore
        groups: int = 1,
        padding_mode: str = "zeros",
        bias: bool = True,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation: torch.nn.Module = torch.nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: tuple = (-1, 1),
        device="cpu",
    ):
        """
        Convolutional layer with KAN kernels. A drop-in replacement for torch.nn.Conv2d.

        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel. Default: 3
            stride (int or tuple): Stride of the convolution. Default: 1
            padding (int, tuple, or str): Padding added to both sides of the input. Default: 0
            dilation (int or tuple): Spacing between kernel elements. Default: 1
            groups (int): Number of blocked connections from input channels to output channels. Default: 1
            padding_mode (str): Padding mode. Default: 'zeros'
            bias (bool): Added for compatibility with torch.nn.Conv2d and does make any effect. Default: True
            grid_size (int): Number of grid points for the spline. Default: 5
            spline_order (int): Order of the spline. Default: 3
            scale_noise (float): Scale of the noise. Default: 0.1
            scale_base (float): Scale of the base. Default: 1.0
            scale_spline (float): Scale of the spline. Default: 1.0
            enable_standalone_scale_spline (bool): Enable standalone scale for the spline. Default: True
            base_activation (torch.nn.Module): Activation function for the base. Default: torch.nn.SiLU
            grid_eps (float): Epsilon for the grid
            grid_range (tuple): Range of the grid. Default: (-1, 1).
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = padding
        self.dilation = _pair(dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        self.device = device
        if isinstance(padding, str):
            if padding == "same":
                self.padding = self._calculate_same_padding()
            elif padding == "valid":
                self.padding = (0, 0)
            else:
                raise ValueError(f"Invalid padding mode: {padding}")
        else:
            self.padding = _pair(padding)

        self._in_dim = (
            (in_channels // groups) * self.kernel_size[0] * self.kernel_size[1]
        )
        self._reversed_padding_repeated_twice = tuple(
            x for x in reversed(self.padding) for _ in range(2)
        )

        if not bias:
            # warn the user that bias is not used
            warnings.warn("Bias is not used in ConvKAN layer", UserWarning)

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        self.kan_layer = KANLinear(
            in_features=self._in_dim,
            out_features=out_channels // groups,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            enable_standalone_scale_spline=enable_standalone_scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
            groups=groups,
            device=device,
        )

    def forward(self, x):
        if self.padding_mode != "zeros":
            x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = (0, 0)  # Reset padding because we already applied it
        else:
            padding = self.padding

        x_unf = F.unfold(
            x,
            kernel_size=self.kernel_size,
            padding=padding,
            stride=self.stride,
            dilation=self.dilation,
        )

        batch_size, channels_and_elem, n_patches = x_unf.shape

        # Ensuring group separation is maintained in the input
        x_unf = (
            x_unf.permute(0, 2, 1)  # [B, H_out * W_out, channels * elems]
            .reshape(
                batch_size * n_patches, self.groups, channels_and_elem // self.groups
            )  # [B * H_out * W_out, groups, out_channels // groups]
            .permute(1, 0, 2)
        )  # [groups, B * H_out * W_out, out_channels // groups]

        output = self.kan_layer(
            x_unf
        )  # [groups, B * H_out * W_out, out_channels // groups]
        output = (
            output.permute(1, 0, 2).reshape(batch_size, n_patches, -1).permute(0, 2, 1)
        )

        # Compute output dimensions
        output_height = (
            x.shape[2]
            + 2 * padding[0]
            - self.dilation[0] * (self.kernel_size[0] - 1)
            - 1
        ) // self.stride[0] + 1
        output_width = (
            x.shape[3]
            + 2 * padding[1]
            - self.dilation[1] * (self.kernel_size[1] - 1)
            - 1
        ) // self.stride[1] + 1

        # Reshape output to the expected output format
        output = output.view(
            x.shape[0],  # batch size
            self.out_channels,  # total output channels
            output_height,
            output_width,
        )

        return output

    def _calculate_same_padding(self):
        padding = []
        for i in range(2):
            pad = self.dilation[i] * (self.kernel_size[i] - 1) // 2
            padding.append(pad)
        return tuple(padding)


def _pair(x):
    if isinstance(x, (int, float)):
        return x, x
    return x


class LayerNorm2D(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x):
        # Compute the mean and variance
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)

        # Normalize
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift
        return self.gamma * x_normalized + self.beta


class KAN_conv2d(nn.Module):
    def __init__(
        self,
        kwargs,
    ):
        super(KAN_conv2d, self).__init__()
        layers_hidden = kwargs.get("layers_hidden")
        kernel_sizes = kwargs.get("kernel_sizes")
        strides = kwargs.get("strides")
        paddings = kwargs.get("paddings")
        grid_size = kwargs.get("grid_size")
        spline_order = kwargs.get("spline_order")
        scale_noise = kwargs.get("scale_noise")
        scale_base = kwargs.get("scale_base")
        scale_spline = kwargs.get("scale_spline")
        base_activation = kwargs.get("base_activation")
        grid_eps = kwargs.get("grid_eps")
        grid_range = kwargs.get("grid_range")
        self.grid_size = kwargs.get("grid_size")
        self.spline_order = kwargs.get("spline_order")
        self.layers = nn.ModuleList()

        for in_channels, out_channels, kernel_size, stride, padding in zip(
            layers_hidden[:-1], layers_hidden[1:], kernel_sizes, strides, paddings
        ):
            self.layers.append(
                KANConv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    grid_size=grid_size,
                    groups=1,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
