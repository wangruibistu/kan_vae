import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
        device=None,
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


def calculate_unfold_shape(input_chennal, input_shape, kernel_size, stride, padding):
    H, W = input_shape
    # print(N, C, H, W)
    unfold_height = (H + 2 * padding - kernel_size) // stride + 1
    unfold_width = (W + 2 * padding - kernel_size) // stride + 1
    unfold_dim = input_chennal * kernel_size * kernel_size
    num_patches = unfold_height * unfold_width

    return unfold_dim, num_patches


def calculate_deconv_output_shape(
    input_shape, kernel_size, stride, padding, output_padding=0
):
    kernel_size = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
    stride = stride[0] if isinstance(stride, tuple) else stride
    padding = padding[0] if isinstance(padding, tuple) else padding
    output_padding = (
        output_padding[0] if isinstance(output_padding, tuple) else output_padding
    )
    H_in, W_in = input_shape
    H_out = (H_in - 1) * stride - 2 * padding + kernel_size + output_padding
    W_out = (W_in - 1) * stride - 2 * padding + kernel_size + output_padding
    return (H_out, W_out)


class KAN_ConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels,
        in_shape,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        bias=True,
        **kwargs,
    ):
        super(KAN_ConvTranspose2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_padding = (
            output_padding
            if isinstance(output_padding, tuple)
            else (output_padding, output_padding)
        )
        self.unfold_in_size = calculate_unfold_shape(
            input_chennal=in_channels,
            input_shape=in_shape,
            kernel_size=self.kernel_size[0],
            padding=self.padding[0],
            stride=self.stride[0],
        )
        self.output_height, self.output_width = calculate_deconv_output_shape(
            in_shape, kernel_size, stride, padding, output_padding
        )
        self.unfold_out_size = calculate_unfold_shape(
            input_chennal=out_channels,
            input_shape=[self.output_height, self.output_width],
            kernel_size=self.kernel_size[0],
            padding=self.padding[0],
            stride=self.stride[0],
        )
        self.kan_linear = KANLinear(
            in_features=math.prod(self.unfold_in_size),
            out_features=math.prod(self.unfold_out_size),
            **kwargs,
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        batch_size, n_channels, h, w = input.size()
        unfold = batched_unfold(
            input,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
            dilation=1,
        )
        del input
        torch.cuda.empty_cache()
        # unfolded_size = unfold.size()
        # print("unfold size: ", unfold.size())
        unfold = unfold.reshape(batch_size, math.prod(self.unfold_in_size))
        # print("unfold size before linear: ", unfold.size())
        unfold = self.kan_linear(unfold)
        # print("unfold size after linear: ", unfold.size())
        unfold = unfold.view(batch_size, self.unfold_out_size[0], self.unfold_out_size[1])
        # print("unfold size before fold: ", unfold.size())
        fold = batched_fold(
            unfold,
            output_size=(self.output_height, self.output_width),
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        )
        del unfold
        torch.cuda.empty_cache()
        if self.bias is not None:
            fold += self.bias.view(1, -1, 1, 1)

        return fold


class KAN_deconv2d(nn.Module):
    def __init__(
        self,
        kwargs,
    ):
        super(KAN_deconv2d, self).__init__()
        layers_hidden = kwargs.get("layers_hidden")
        input_shape = kwargs.get("input_shape")
        kernel_sizes = kwargs.get("kernel_sizes")
        strides = kwargs.get("strides")
        paddings = kwargs.get("paddings")
        output_paddings = kwargs.get("output_paddings")
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

        for (
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
        ) in zip(
            layers_hidden[:-1],
            layers_hidden[1:],
            kernel_sizes,
            strides,
            paddings,
            output_paddings,
        ):
            self.layers.append(
                KAN_ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    in_shape=input_shape,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                )
            )
            input_shape = convtranspose2d_output_shape(
                input_shape[0],
                input_shape[1],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                # dilation=1
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
    

def batched_unfold(input, batch_size, kernel_size, padding, stride):
    unfold_outputs = []
    for i in range(0, input.size(0), batch_size):
        batch_input = input[i:i+batch_size]
        unfold_output = F.unfold(
            batch_input,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        unfold_outputs.append(unfold_output)
    return torch.cat(unfold_outputs, dim=0)

def batched_fold(input, output_size, kernel_size, padding, stride, batch_size):
    fold_outputs = []
    for i in range(0, input.size(0), batch_size):
        batch_input = input[i:i+batch_size]
        fold_output = F.fold(
            batch_input,
            output_size=output_size,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        fold_outputs.append(fold_output)
    return torch.cat(fold_outputs, dim=0)


def conv2d_output_shape(h, w, kernel_size=1, stride=1, padding=0, dilation=1):
    # if type(h_w) is not tuple:
    # h_w = (h_w, h_w)
    h_out = (h + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    w_out = (w + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    return h_out, w_out


def convtranspose2d_output_shape(
    h, w, kernel_size=1, stride=1, padding=0, output_padding=0, dilation=1
):
    h_out = (
        (h - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + output_padding
        + 1
    )
    w_out = (
        (w - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + output_padding
        + 1
    )
    return h_out, w_out


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        device=None,
    ):
        super(KANConvTranspose2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
        self.groups = groups
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]).expand(in_channels, -1).contiguous()
        self.register_buffer("grid", grid)
        
        self.base_weight = nn.Parameter(torch.Tensor(in_channels, out_channels // groups, *self.kernel_size))
        self.spline_weight = nn.Parameter(torch.Tensor(in_channels, out_channels // groups, *self.kernel_size, grid_size + spline_order))
        
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(in_channels, out_channels // groups, *self.kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (torch.rand(*self.spline_weight.shape) - 1/2) * self.scale_noise / self.grid_size
            self.spline_weight.data.copy_((self.scale_spline if not self.enable_standalone_scale_spline else 1.0) * noise)
            
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.base_weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def b_splines(self, x):
        assert x.dim() == 4 and x.size(1) == self.in_channels
        
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[:, :-(k+1)]) / (grid[:, k:-1] - grid[:, :-(k+1)]) * bases[:, :, :, :, :-1]) + \
                    ((grid[:, k+1:] - x) / (grid[:, k+1:] - grid[:, 1:(-k)]) * bases[:, :, :, :, 1:])
        
        return bases.contiguous()
    
    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0)
    
    def forward(self, x):
        base_output = F.conv_transpose2d(
            self.base_activation(x),
            self.base_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups,
            dilation=self.dilation
        )
        
        spline_bases = self.b_splines(x)
        spline_output = F.conv_transpose2d(
            spline_bases.view(x.size(0), -1, *x.shape[2:]),
            self.scaled_spline_weight.view(self.in_channels * self.out_channels // self.groups, -1, *self.kernel_size),
            bias=None,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups * self.in_channels,
            dilation=self.dilation
        )
        spline_output = spline_output.view(x.size(0), self.out_channels, *spline_output.shape[2:])
        
        return base_output + spline_output