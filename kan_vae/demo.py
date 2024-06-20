import torch
from torch import nn
import torch.nn.functional as F
import math
from kan_model import KANLinear, KANConv2d


def _pair(x):
    if isinstance(x, (int, float)):
        return x, x
    return x


class KANConvTranspose2d(torch.nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        bias=True,
    ):
        super(KANConvTranspose2d, self).__init__()
        self.in_channels = in_channel
        self.out_channels = out_channel
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
        self.weight = torch.nn.Parameter(
            torch.Tensor(in_channel, out_channel, *self.kernel_size)
        )
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channel))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        output_height = (
            (input.size(2) - 1) * self.stride[0]
            - 2 * self.padding[0]
            + self.kernel_size[0]
            + self.output_padding[0]
        )
        output_width = (
            (input.size(3) - 1) * self.stride[1]
            - 2 * self.padding[1]
            + self.kernel_size[1]
            + self.output_padding[1]
        )

        output = torch.zeros(
            (input.size(0), self.out_channels, output_height, output_width),
            device=input.device,
            dtype=input.dtype,
        )

        kan_conv2d = KANConv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=1,
            padding=self.padding,
        )

        for i in range(input.size(2)):
            for j in range(input.size(3)):
                output[
                    :,
                    :,
                    i * self.stride[0] : i * self.stride[0] + self.kernel_size[0],
                    j * self.stride[1] : j * self.stride[1] + self.kernel_size[1],
                ] += kan_conv2d(input[:, :, i : i + 1, j : j + 1])

        return output






if __name__ == "__main__":
    batch_size = 1
    in_channels = 64
    height = 16
    width = 16
    # model = KANConvTranspose2d(
    #     in_channels=in_channels,
    #     out_channels=1,
    #     kernel_size=3,
    #     stride=2,
    #     padding=1,
    #     output_padding=1,
    #     grid_size=5,
    #     spline_order=3,
    #     scale_noise=0.1,
    #     scale_base=1.0,
    #     scale_spline=1.0,
    #     enable_standalone_scale_spline=True,
    #     base_activation=nn.SiLU,
    #     grid_eps=0.02,
    #     grid_range=(-1, 1),
    # )

    x = torch.randn(batch_size, in_channels, height, width)

    # output = model(x)
    # print(f"Input shape: {x.shape}")
    # print(f"Output shape: {output.shape}")
    custom_conv_transpose2d = KANConvTranspose2d(
        in_channel=64,
        out_channel=16,
        kernel_size=3,
        stride=2,
        padding=1,
        output_padding=1,
    )

    # 计算输出
    output_tensor = custom_conv_transpose2d(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_tensor.shape}")
