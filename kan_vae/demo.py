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
        device=None
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


class KANConvTranspose2d(nn.Module):
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
        super(KANConvTranspose2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
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

    # def reset_parameters(self):
    #     nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    #     if self.bias is not None:
    #         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
    #         bound = 1 / math.sqrt(fan_in)
    #         nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        unfold = F.unfold(
            input,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
            dilation=1,
        )
        # unfolded_size = unfold.size()

        # 调整形状以适应KANLinear
        print('unfold size: ', unfold.size())
        unfold = unfold.reshape(-1, math.prod(self.unfold_in_size))
        print("unfold size before linear: ", unfold.size())
        unfold = self.kan_linear(unfold)
        print("unfold size after linear: ", unfold.size())

        unfold = unfold.view(-1, self.unfold_out_size[0], self.unfold_out_size[1])

        print("unfold size before fold: ", unfold.size())
        print(self.kernel_size, self.padding, self.stride)
        fold = F.fold(
            unfold,
            output_size=(self.output_height, self.output_width),
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        )

        if self.bias is not None:
            fold += self.bias.view(1, -1, 1, 1)

        return fold


class KAN_ConvTranspose_Layer(torch.nn.Module):
    def __init__(
        self,
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

        super(KAN_ConvTranspose_Layer, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.kernel_size = kernel_size
        self.device = device
        self.dilation = dilation
        self.padding = padding
        self.convs = torch.nn.ModuleList()
        self.n_convs = n_convs
        self.stride = stride

        # Create n_convs KAN_Convolution objects
        for _ in range(n_convs):
            self.convs.append(
                KANConvTranspose2d(
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
            )
            # 测试代码
            # kan_conv_transpose_layer = KANConvTranspose2d(
            #     in_channels=3,
            #     in_shape=(16, 16),
            #     out_channels=2,
            #     kernel_size=(3, 3),
            #     stride=(2, 2),
            #     padding=(1, 1),
            #     output_padding=(1, 1),
            #     grid_size=5,
            #     spline_order=3,
            #     scale_noise=0.1,
            #     scale_base=1.0,
            #     scale_spline=1.0,
            #     base_activation=nn.SiLU,
            #     grid_eps=0.02,
            #     grid_range=(-1, 1),
            #     device="cpu",
            # )

            # input_tensor = torch.randn(8, 3, 16, 16, device="cpu")
            # output_tensor = kan_conv_transpose_layer(input_tensor)
            # print(f"Input shape: {input_tensor.shape}")
            # print(f"Output shape: {output_tensor.shape}")
    def forward(self, x: torch.Tensor, update_grid=False):
        # If there are multiple convolutions, apply them all
        if self.n_convs > 1:
            return multiple_convs_kan_conv2d(
                x,
                self.convs,
                self.kernel_size[0],
                self.stride,
                self.dilation,
                self.padding,
                self.device,
            )

        # If there is only one convolution, apply it
        return self.convs[0].forward(x)
