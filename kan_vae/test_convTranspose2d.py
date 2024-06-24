import torch
import torch.nn as nn
import torch.nn.functional as F

from kan_conv_transpose2d import KANConvTranspose2d


def test_kan_conv_transpose2d():
    # 设置随机种子以确保结果可复现
    torch.manual_seed(0)

    # 定义输入参数
    batch_size = 2
    in_channels = 3
    out_channels = 1
    input_height = 32
    input_width = 32
    kernel_size = 3
    stride = 2
    padding = 1
    output_padding = 1

    # 创建 KANConvTranspose2d 实例
    kan_conv_transpose = KANConvTranspose2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        bias=True,
    )

    # 创建随机输入数据
    x = torch.randn(batch_size, in_channels, input_height, input_width)

    # 前向传播
    output = kan_conv_transpose(x)

    # 检查输出形状
    expected_output_height = (
        (input_height - 1) * stride - 2 * padding + kernel_size + output_padding
    )
    expected_output_width = (
        (input_width - 1) * stride - 2 * padding + kernel_size + output_padding
    )
    expected_output_shape = (
        batch_size,
        out_channels,
        expected_output_height,
        expected_output_width,
    )

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: {expected_output_shape}")

    assert (
        output.shape == expected_output_shape
    ), f"Output shape {output.shape} does not match expected shape {expected_output_shape}"

    # 检查输出值
    print(f"Output mean: {output.mean().item():.4f}")
    print(f"Output std: {output.std().item():.4f}")

    # 检查梯度
    output.sum().backward()

    print(
        f"Base weight grad mean: {kan_conv_transpose.base_weight.grad.mean().item():.4f}"
    )
    print(
        f"Spline weight grad mean: {kan_conv_transpose.spline_weight.grad.mean().item():.4f}"
    )

    if kan_conv_transpose.enable_standalone_scale_spline:
        print(
            f"Spline scaler grad mean: {kan_conv_transpose.spline_scaler.grad.mean().item():.4f}"
        )

    print(f"Bias grad mean: {kan_conv_transpose.bias.grad.mean().item():.4f}")

    print("All tests passed!")


# 运行测试
test_kan_conv_transpose2d()
