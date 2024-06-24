import torch
import math

# from KANLinear import KANLinear
# import convolution
import torch
import numpy as np
from typing import List, Tuple, Union
from torch import nn
import torch.nn.functional as F
import math
import warnings


def _pair(param):
    if isinstance(param, tuple):
        return param
    return (param, param)


def calc_out_dims(matrix, kernel_side, stride, dilation, padding):
    batch_size, n_channels, n, m = matrix.shape
    h_out = (
        np.floor(
            (
                n
                + 2 * padding[0]
                - kernel_side[0]
                - (kernel_side[0] - 1) * (dilation[0] - 1)
            )
            / stride[0]
        ).astype(int)
        + 1
    )
    w_out = (
        np.floor(
            (
                m
                + 2 * padding[1]
                - kernel_side[1]
                - (kernel_side[1] - 1) * (dilation[1] - 1)
            )
            / stride[1]
        ).astype(int)
        + 1
    )
    b = [kernel_side[0] // 2, kernel_side[1] // 2]
    return h_out, w_out, batch_size, n_channels


def kan_conv2d(
    matrix,  # but as torch tensors. Kernel side asume q el kernel es cuadrado
    kernel,
    kernel_side,
    stride=(1, 1),
    dilation=(1, 1),
    padding=(0, 0),
    device="cuda",
) -> torch.Tensor:
    """Makes a 2D convolution with the kernel over matrix using defined stride, dilation and padding along axes.

    Args:
        matrix (batch_size, colors, n, m]): 2D matrix to be convolved.
        kernel  (function]): 2D odd-shaped matrix (e.g. 3x3, 5x5, 13x9, etc.).
        stride (Tuple[int, int], optional): Tuple of the stride along axes. With the `(r, c)` stride we move on `r` pixels along rows and on `c` pixels along columns on each iteration. Defaults to (1, 1).
        dilation (Tuple[int, int], optional): Tuple of the dilation along axes. With the `(r, c)` dilation we distancing adjacent pixels in kernel by `r` along rows and `c` along columns. Defaults to (1, 1).
        padding (Tuple[int, int], optional): Tuple with number of rows and columns to be padded. Defaults to (0, 0).

    Returns:
        np.ndarray: 2D Feature map, i.e. matrix after convolution.
    """
    # kernel_size = _pair(kernel_size)
    # stride = _pair(stride)
    # dilation = _pair(dilation)
    # padding = _pair(padding)
    h_out, w_out, batch_size, n_channels = calc_out_dims(
        matrix, kernel_side, stride, dilation, padding
    )

    matrix_out = torch.zeros((batch_size, n_channels, h_out, w_out)).to(
        device
    )  # estamos asumiendo que no existe la dimension de rgb
    unfold = torch.nn.Unfold(
        kernel_size=kernel_side,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )

    for channel in range(n_channels):
        conv_groups = unfold(matrix[:, channel, :, :].unsqueeze(1)).transpose(1, 2)
        for k in range(batch_size):
            matrix_out[k, channel, :, :] = kernel.forward(conv_groups[k, :, :]).view(
                (h_out, w_out)
            )
    return matrix_out


def multiple_convs_kan_conv2d(
    matrix,  # but as torch tensors. Kernel side asume q el kernel es cuadrado
    kernels,
    kernel_side,
    stride=(1, 1),
    dilation=(1, 1),
    padding=(0, 0),
    device="cuda",
) -> torch.Tensor:
    """Makes a 2D convolution with the kernel over matrix using defined stride, dilation and padding along axes.

    Args:
        matrix (batch_size, colors, n, m]): 2D matrix to be convolved.
        kernel  (function]): 2D odd-shaped matrix (e.g. 3x3, 5x5, 13x9, etc.).
        stride (Tuple[int, int], optional): Tuple of the stride along axes. With the `(r, c)` stride we move on `r` pixels along rows and on `c` pixels along columns on each iteration. Defaults to (1, 1).
        dilation (Tuple[int, int], optional): Tuple of the dilation along axes. With the `(r, c)` dilation we distancing adjacent pixels in kernel by `r` along rows and `c` along columns. Defaults to (1, 1).
        padding (Tuple[int, int], optional): Tuple with number of rows and columns to be padded. Defaults to (0, 0).

    Returns:
        np.ndarray: 2D Feature map, i.e. matrix after convolution.
    """
    h_out, w_out, batch_size, n_channels = calc_out_dims(
        matrix, kernel_side, stride, dilation, padding
    )
    n_convs = len(kernels)
    matrix_out = torch.zeros((batch_size, n_channels * n_convs, h_out, w_out)).to(
        device
    )  # estamos asumiendo que no existe la dimension de rgb
    unfold = torch.nn.Unfold(
        (kernel_side, kernel_side), dilation=dilation, padding=padding, stride=stride
    )
    conv_groups = (
        unfold(matrix[:, :, :, :])
        .view(batch_size, n_channels, kernel_side * kernel_side, h_out * w_out)
        .transpose(2, 3)
    )  # reshape((batch_size,n_channels,h_out,w_out))
    for channel in range(n_channels):
        for kern in range(n_convs):
            matrix_out[:, kern + channel * n_convs, :, :] = (
                kernels[kern]
                .conv.forward(conv_groups[:, channel, :, :].flatten(0, 1))
                .reshape((batch_size, h_out, w_out))
            )
    return matrix_out


def add_padding(matrix: np.ndarray, padding: Tuple[int, int]) -> np.ndarray:
    """Adds padding to the matrix.

    Args:
        matrix (np.ndarray): Matrix that needs to be padded. Type is List[List[float]] casted to np.ndarray.
        padding (Tuple[int, int]): Tuple with number of rows and columns to be padded. With the `(r, c)` padding we addding `r` rows to the top and bottom and `c` columns to the left and to the right of the matrix

    Returns:
        np.ndarray: Padded matrix with shape `n + 2 * r, m + 2 * c`.
    """
    n, m = matrix.shape
    r, c = padding

    padded_matrix = np.zeros((n + r * 2, m + c * 2))
    padded_matrix[r : n + r, c : m + c] = matrix

    return padded_matrix


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(
            self.base_weight, a=math.sqrt(5) * self.scale_base
        )
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline
                )

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        print(x.shape)
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
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


class KAN_Convolution(torch.nn.Module):
    def __init__(
        self,
        in_channel=None,
        out_channel=None,
        kernel_size: tuple = (2, 2),
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation=torch.nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: tuple = [-1, 1],
        device="cpu",
    ):
        """
        Args
        """
        super(KAN_Convolution, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.device = device
        self.conv = KANLinear(
            in_features=math.prod(kernel_size) * in_channel,
            out_features=out_channel,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
        )

    def forward(self, x: torch.Tensor, update_grid=False):
        return kan_conv2d(
            x,
            self.conv,
            self.kernel_size,
            self.stride,
            self.dilation,
            self.padding,
            self.device,
        )

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )


# Script que contiene la implementación del kernel con funciones de activación.
class KAN_Convolutional_Layer(torch.nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        n_convs: int = 1,
        kernel_size: tuple = (2, 2),
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation=torch.nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: tuple = [-1, 1],
        device: str = "cpu",
    ):
        """
        Kan Convolutional Layer with multiple convolutions

        Args:
            n_convs (int): Number of convolutions to apply
            kernel_size (tuple): Size of the kernel
            stride (tuple): Stride of the convolution
            padding (tuple): Padding of the convolution
            dilation (tuple): Dilation of the convolution
            grid_size (int): Size of the grid
            spline_order (int): Order of the spline
            scale_noise (float): Scale of the noise
            scale_base (float): Scale of the base
            scale_spline (float): Scale of the spline
            base_activation (torch.nn.Module): Activation function
            grid_eps (float): Epsilon of the grid
            grid_range (tuple): Range of the grid
            device (str): Device to use
        """

        super(KAN_Convolutional_Layer, self).__init__()

        self.grid_size = grid_size
        self.spline_order = spline_order
        self.kernel_size = _pair(kernel_size)
        self.device = device
        self.dilation = _pair(dilation)
        self.padding = _pair(padding)
        self.convs = torch.nn.ModuleList()
        self.n_convs = n_convs
        self.stride = _pair(stride)

        # Create n_convs KAN_Convolution objects
        # for _ in range(n_convs):
        self.convs = KAN_Convolution(
                    in_channel=in_channel,
                    out_channel=out_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                    device=device,
                )

    def forward(self, x: torch.Tensor, update_grid=False):
        # If there are multiple convolutions, apply them all
        # if self.n_convs > 1:
        #     return multiple_convs_kan_conv2d(
        #         x,
        #         self.convs,
        #         self.kernel_size[0],
        #         self.stride,
        #         self.dilation,
        #         self.padding,
        #         self.device,
        #     )

        # If there is only one convolution, apply it
        return self.convs.forward(x)

# 示例输入
input_tensor = torch.randn(1, 3, 32, 32).to("cuda")

# 定义卷积层
conv_layer = KAN_Convolutional_Layer(
    in_channel=3,
    out_channel=9,
    kernel_size=(3, 3),
    stride=(1, 1),
    padding=(1, 1),
    grid_size=5,
    spline_order=3,
    scale_noise=0.1,
    scale_base=1.0,
    scale_spline=1.0,
    base_activation=torch.nn.SiLU,
    grid_eps=0.02,
    grid_range=[-1, 1],
    device="cuda",
)

# 将卷积层移动到CUDA
conv_layer.to("cuda")

# 前向传播
output_tensor = conv_layer(input_tensor)

print("输入形状:", input_tensor.shape)
print("输出形状:", output_tensor.shape)
