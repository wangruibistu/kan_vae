import torch
from torch import nn
import torch.nn.functional as F
import math
import warnings


class KANLinear1D(torch.nn.Module):
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
        device="cuda",
    ):
        super(KANLinear1D, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.device = device
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
            .to(device)
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features).to(device)
        )
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order).to(device)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features).to(device)
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
                    torch.rand(
                        self.grid_size + 1, self.in_features, self.out_features
                    ).to(self.device)
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
        groups: int = 1,
        device="cpu",
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.groups = groups
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(
                    -spline_order,
                    grid_size + spline_order + 1,
                )
                * h
                + grid_range[0]
            )
            .expand(groups, in_features, -1)
            .contiguous()
        ).to(device)
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(
            torch.Tensor(
                groups,
                out_features,
                in_features,
            ).to(device)
        )
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(
                groups,
                out_features,
                in_features,
                grid_size + spline_order,
            ).to(device)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(
                    groups,
                    out_features,
                    in_features,
                ).to(device)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(
            self.base_weight, a=math.sqrt(5) * self.scale_base
        )
        with torch.no_grad():
            noise = (
                (
                    torch.rand(
                        self.groups,
                        self.grid_size + 1,
                        self.in_features,
                        self.out_features,
                        device=self.device,
                    )
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.transpose(-1, -2)[
                        :, self.spline_order : -self.spline_order
                    ],
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
        assert x.dim() == 3 and x.size(2) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (groups, in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        grid = grid.unsqueeze(1)

        bases = ((x >= grid[:, :, :, :-1]) & (x < grid[:, :, :, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, :, :, : -(k + 1)])
                / (grid[:, :, :, k:-1] - grid[:, :, :, : -(k + 1)])
                * bases[:, :, :, :-1]
            ) + (
                (grid[:, :, :, k + 1 :] - x)
                / (grid[:, :, :, k + 1 :] - grid[:, :, :, 1:(-k)])
                * bases[:, :, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            x.size(1),
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
        assert x.dim() == 3 and x.size(2) == self.in_features
        assert y.size() == (x.size(0), x.size(1), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            1, 2
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(1, 2)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            0, 3, 1, 2
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.groups,
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
        # print(x.shape, self.in_features)
        assert x.dim() == 3 and x.size(2) == self.in_features

        base_output = torch.bmm(
            self.base_activation(x), self.base_weight.permute(0, 2, 1)
        )
        spline_output = torch.bmm(
            self.b_splines(x).view(x.size(0), x.size(1), -1),
            self.scaled_spline_weight.view(x.size(0), self.out_features, -1).permute(
                0, 2, 1
            ),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 3 and x.size(2) == self.in_features
        batch = x.size(1)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(0, 2, 1, 3)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(0, 2, 3, 1)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            0, 2, 1, 3
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=1)[0]
        grid_adaptive = x_sorted[
            :,
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            ),
        ]

        uniform_step = (x_sorted[:, -1] - x_sorted[:, 0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(self.grid_size + 1, dtype=torch.float32, device=x.device)
            .unsqueeze(1)
            .unsqueeze(0)
            * uniform_step.unsqueeze(1)
            + x_sorted[:, 0].unsqueeze(1)
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:, :1]
                - uniform_step.unsqueeze(1)
                * torch.arange(self.spline_order, 0, -1, device=x.device)
                .unsqueeze(1)
                .unsqueeze(0),
                grid,
                grid[:, -1:]
                + uniform_step.unsqueeze(1)
                * torch.arange(1, self.spline_order + 1, device=x.device)
                .unsqueeze(1)
                .unsqueeze(0),
            ],
            dim=1,
        )

        self.grid.copy_(grid.transpose(-1, -2))
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
                KANLinear1D(
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

    def save_model(self, name):
        torch.save()

