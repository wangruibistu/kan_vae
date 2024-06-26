import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _pair(param):
    if isinstance(param, tuple):
        return param
    return (param, param)


class KANConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        grid_size=3,
        spline_order=3,
        enable_standalone_scale_spline=True,
        # device="cuda"
    ):
        super(KANConvTranspose2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.output_padding = _pair(output_padding)
        self.groups = groups
        self.dilation = _pair(dilation)
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.enable_standalone_scale_spline = enable_standalone_scale_spline

        self.base_weight = nn.Parameter(
            torch.Tensor(
                in_channels,
                out_channels // groups,
                *self.kernel_size,
            )#.to(device)
        )
        self.spline_weight = nn.Parameter(
            torch.Tensor(
                in_channels,
                out_channels // groups,
                *self.kernel_size,
                grid_size + spline_order,
            )#.to(device)
        )

        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.Tensor(
                    in_channels,
                    out_channels // groups,
                    *self.kernel_size,
                )#.to(device)
            )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))#.to(device))
        else:
            self.register_parameter("bias", None)

        self.grid = nn.Parameter(
            torch.linspace(
                -1,
                1,
                grid_size + 2 * spline_order + 1,
            ),#.to(device),
            requires_grad=False,
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5))
        if self.enable_standalone_scale_spline:
            nn.init.constant_(self.spline_scaler, 1.0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.base_weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def base_activation(self, x):
        return x

    def b_splines(self, x):
        assert x.dim() == 4 and x.size(1) == self.in_channels

        grid = self.grid  # shape: (grid_size + 2 * spline_order + 1)
        x_norm = (x - x.min()) / (x.max() - x.min()) * 2 - 1  # Normalize x to [-1, 1]

        # Reshape x for broadcasting
        x_flat = x_norm.view(
            x.size(0), x.size(1), -1, 1
        )  # [batch, channels, height*width, 1]

        # Expand grid for broadcasting
        grid_expanded = grid.view(1, 1, 1, -1).expand(
            x_flat.size(0), x_flat.size(1), x_flat.size(2), -1
        )

        bases = (
            (x_flat >= grid_expanded[..., :-1]) & (x_flat < grid_expanded[..., 1:])
        ).float()

        for k in range(1, self.spline_order + 1):
            weights = (x_flat - grid_expanded[..., : -(k + 1)]) / (
                grid_expanded[..., k:-1] - grid_expanded[..., : -(k + 1)]
            )
            bases_left = weights * bases[..., :-1]

            weights = (grid_expanded[..., k + 1 :] - x_flat) / (
                grid_expanded[..., k + 1 :] - grid_expanded[..., 1:-k]
            )
            bases_right = weights * bases[..., 1:]

            bases = bases_left + bases_right

        # Reshape bases back to match input spatial dimensions
        bases = bases.view(x.size(0), x.size(1), x.size(2), x.size(3), -1)

        return bases

    def forward(self, x):
        base_output = F.conv_transpose2d(
            self.base_activation(x),
            self.base_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups,
            dilation=self.dilation,
        )

        spline_bases = self.b_splines(x)

        # Reshape spline_bases for convolution
        x_spline = x.unsqueeze(-1) * spline_bases
        x_spline = x_spline.view(x.size(0), -1, *x.shape[2:])

        # Reshape scaled_spline_weight for convolution
        scaled_spline_weight = self.scaled_spline_weight.view(
            self.in_channels, self.out_channels // self.groups, -1, *self.kernel_size
        )
        scaled_spline_weight = scaled_spline_weight.reshape(
            -1, x_spline.size(1), *self.kernel_size
        ).transpose(
            1, 0
        ) 
        spline_output = F.conv_transpose2d(
            x_spline,
            scaled_spline_weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups,
            dilation=self.dilation,
        )

        return base_output + spline_output


class KAN_ConvTranspose_Layer(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        grid_size=3,
        spline_order=3,
        enable_standalone_scale_spline=True,
        # device="cuda"
    ):
        super(KAN_ConvTranspose_Layer, self).__init__()

        self.trans_conv = KANConvTranspose2d(
            in_channel,
            out_channel,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            grid_size,
            spline_order,
            enable_standalone_scale_spline,
            # device=device
        )

    def forward(self, x):
        return self.trans_conv(x)
