import torch
import torch.nn as nn
import torch.nn.functional as F

from kan_linear import KANLinear, KANLinear1D
from kan_conv import KANConv2d # KAN_deconv2d

from kan_conv_transpose2d import KAN_ConvTranspose_Layer


# class KAN_VAE2D(nn.Module):
#     def __init__(
#         self,
#         input_channels,
#         encoder_hidden_dims,
#         decoder_hidden_dims,
#         latent_dim,
#         input_shape=(28, 28),
#         encoderkwargs=None,
#         decoderkwargs=None,
#     ):
#         super(KAN_VAE2D, self).__init__()
#         self.encoder_hidden_dims = encoder_hidden_dims
#         self.decoder_hidden_dims = decoder_hidden_dims
#         encoderkwargs["layers_hidden"] = [input_channels] + encoder_hidden_dims
#         decoderkwargs["layers_hidden"] = decoder_hidden_dims + [input_channels]
#         decoderkwargs["input_shape"] = input_shape

#         self.encoder = KAN_encoder(encoderkwargs)

#         self.flatten = nn.Flatten()

#         encoder_output_shape = calculate_encoder_output_shape(
#             self.encoder, (input_channels, *input_shape)
#         )
#         self.flattened_size = encoder_output_shape

#         self.fc_mu = nn.Linear(
#             encoder_output_shape[0] * encoder_output_shape[1] * encoder_output_shape[2],
#             latent_dim,
#         ).to(device=device)

#         self.fc_var = nn.Linear(
#             encoder_output_shape[0] * encoder_output_shape[1] * encoder_output_shape[2],
#             latent_dim,
#         ).to(device=device)

#         # self.fc_mu = nn.Conv2d(encoder_hidden_dims[-1], latent_dim, kernel_size=1)
#         # self.fc_var = nn.Conv2d(encoder_hidden_dims[-1], latent_dim, kernel_size=1)

#         self.decoder_input = nn.Linear(
#             latent_dim,
#             encoder_output_shape[0] * encoder_output_shape[1] * encoder_output_shape[2],
#         ).to(device=device)

#         self.decoder = nn.Sequential(
#             KAN_deconv2d(decoderkwargs),
#             nn.GELU(),
#         ).to(device=device)

#     def encode(self, x):
#         # print("encoder input: ", x.shape)
#         x = self.encoder(x)
#         # print("encoder conv output: ", x.shape)
#         x = self.flatten(x)
#         # print("encoder flatten output: ", x.shape)
#         mu = self.fc_mu(x)
#         log_var = self.fc_var(x)
#         # print("encoder output: ", mu.shape, log_var.shape)
#         return mu, log_var

#     def reparameterize(self, mu, log_var):
#         std = torch.exp(0.5 * log_var)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def decode(self, z):
#         # print("decoder input shape0:", z.shape)
#         x = self.decoder_input(z)
#         # print("decoder input shape1:", x.shape)
#         x = x.view(
#             -1,
#             self.flattened_size[0],
#             self.flattened_size[1],
#             self.flattened_size[2],
#         )
#         # print("decoder input shape2:", x.shape)
#         x = self.decoder(x)
#         # print("decoder output: ", x.shape)
#         return x

#     def forward(self, x):
#         mu, log_var = self.encode(x)
#         z = self.reparameterize(mu, log_var)
#         reconstructed = self.decode(z)
#         return reconstructed, mu, log_var

#     def loss_function(self, reconstructed, x, mu, log_var):
#         recon_loss = F.mse_loss(reconstructed, x, reduction="sum")
#         kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
#         return recon_loss + kld_loss


def calculate_encoder_output_shape(encoder, input_shape):
    with torch.no_grad():
        dummy_input = torch.zeros(1, *input_shape)
        output = encoder(dummy_input)
        return output.shape[1:]


class KAN_VAE_model(nn.Module):
    def __init__(
        self,
        # input_channels=None,
        # encoder_hidden_dims=None,
        # decoder_hidden_dims=None,
        input_channels=1,
        input_shape=(28, 28),
        latent_dim=1,
        encoderkwargs=None,
        decoderkwargs=None,
        # device="cuda"
    ):
        super(KAN_VAE_model, self).__init__()

        self.encoder = KAN_encoder(kwargs=encoderkwargs)#.to(device)
        self.flatten = nn.Flatten()#.to(device=device)
        self.encoder_output_shape = compute_encoder_output_shape(
            self.encoder, 
            input_channels, 
            input_shape, 
            # device=device,
        )
        self.flattened_size = (
            self.encoder_output_shape[0]
            * self.encoder_output_shape[1]
            * self.encoder_output_shape[2]
        )
        self.fc_mu = KANLinear1D(
            self.flattened_size,
            latent_dim,
            # device=device
        )
        self.fc_var = KANLinear1D(
            self.flattened_size,
            latent_dim,
            # device=device
        )
        self.decoder_input = KANLinear1D(
            latent_dim, 
            self.flattened_size,
            # device=device
        )
        self.decoder = KAN_decoder(decoderkwargs)#.to(device=device)

    def encode(self, x):
        # print("encoder input: ", x.shape)
        x = self.encoder(x)
        # print("encoder conv output: ", x.shape)
        x = self.flatten(x)
        # print("encoder flatten output: ", x.shape)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        # print("encoder output: ", mu.shape, log_var.shape)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # print("decoder input shape0:", z.shape)
        x = self.decoder_input(z)
        # print("decoder input shape1:", x.shape)
        x = x.view(
            -1,
            self.encoder_output_shape[0],
            self.encoder_output_shape[1],
            self.encoder_output_shape[2],
        )
        # print("decoder input shape2:", x.shape)
        x = self.decoder(x)
        # print("decoder output: ", x.shape)
        return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode(z)
        return reconstructed, mu, log_var

    def loss_function(self, reconstructed, x, mu, log_var):
        recon_loss = F.mse_loss(reconstructed, x, reduction="sum")
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + kld_loss


def compute_encoder_output_shape(encoder, channel, input_shape):
    with torch.no_grad():
        dummy_input = torch.randn(1, channel, *input_shape)#.to(device)
        # print(dummy_input.shape)
        output = encoder(dummy_input)
        return output.shape[1:]


class KAN_encoder(nn.Module):
    def __init__(
        self,
        kwargs,
    ):
        super(KAN_encoder, self).__init__()
        layers_hidden = kwargs.get("layers_hidden")
        # input_shape = kwargs.get("input_shape")
        kernel_sizes = kwargs.get("kernel_sizes")
        strides = kwargs.get("strides")
        paddings = kwargs.get("paddings")
        # output_paddings = kwargs.get("output_paddings")
        grid_size = kwargs.get("grid_size")
        spline_order = kwargs.get("spline_order")
        scale_noise = kwargs.get("scale_noise")
        scale_base = kwargs.get("scale_base")
        scale_spline = kwargs.get("scale_spline")
        base_activation = kwargs.get("base_activation")
        grid_eps = kwargs.get("grid_eps")
        grid_range = kwargs.get("grid_range")
        # device = kwargs.get("device")
        # grid_size = kwargs.get("grid_size")
        # spline_order = kwargs.get("spline_order")
        self.layers = nn.ModuleList()

        for (
            in_channel,
            out_channel,
            kernel_size,
            stride,
            padding,
        ) in zip(
            layers_hidden[:-1],
            layers_hidden[1:],
            kernel_sizes,
            strides,
            paddings,
        ):
            self.layers.append(
                KANConv2d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=kernel_size,
                    # in_shape=input_shape,
                    stride=stride,
                    padding=padding,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                    # device=device
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


class KAN_decoder(nn.Module):
    def __init__(
        self,
        kwargs,
    ):
        super(KAN_decoder, self).__init__()
        layers_hidden = kwargs.get("layers_hidden")
        kernel_sizes = kwargs.get("kernel_sizes")
        strides = kwargs.get("strides")
        paddings = kwargs.get("paddings")
        output_paddings = kwargs.get("output_paddings")
        groups = kwargs.get("groups")
        grid_size = kwargs.get("grid_size")
        spline_order = kwargs.get("spline_order")
        dilation = kwargs.get("dilation")
        # device = kwargs.get("device")
        
        self.layers = nn.ModuleList()
        for (
            in_channel,
            out_channel,
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
                KAN_ConvTranspose_Layer(
                    in_channel=in_channel,
                    out_channel=out_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    groups=groups,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    dilation=dilation,
                    # device=device
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


if __name__ == "__main__":
    # device = torch.device('cuda')
    ec_kwargs = {
        "layers_hidden": [1, 16, 32], 
        "kernel_sizes": [(3, 3), (3, 3)], 
        "strides": [(2, 2), (2, 2)], 
        "paddings": [(1, 1), (1, 1)], 
        "grid_size": 5,
        "spline_order": 3,
        "scale_noise": 0.1,
        "scale_base": 1.0,
        "scale_spline": 1.0,
        "base_activation": torch.nn.SiLU,
        "grid_eps": 0.02,
        "grid_range": [-1, 1],
        # "device": device
    }
    dc_kwargs = {
        "layers_hidden": [32, 16, 1],
        "kernel_sizes": [(3, 3), (3, 3)],
        "strides": [(2, 2), (2, 2)],
        "paddings": [(1, 1), (1, 1)],
        "output_paddings": [(1, 1), (1, 1)],
        "groups": 1,
        "grid_size": 5,
        "spline_order": 3,
        "dilation": (1, 1),
        # "device": device,
    }
    # encoder = KAN_encoder(ec_kwargs)

    input_tensor1 = torch.randn(1, 1, 28, 28)#.to(device=device)
    # output = encoder(input_tensor1)
    # print("Encoder输入形状:", input_tensor1.shape)
    # print("Encoder输出形状:", output.shape)

    # decoder = KAN_decoder(dc_kwargs).to(device=device)
    # input_tensor2 = torch.randn(1, 32, 8, 8).to(device=device)
    # output_tensor = decoder(input_tensor2)

    # print("Decoder输入形状:", input_tensor2.shape)
    # print("Decoder输出形状:", output_tensor.shape)

    model = KAN_VAE_model(
        input_channels=1,
        input_shape=(28, 28),
        latent_dim=16,
        encoderkwargs=ec_kwargs,
        decoderkwargs=dc_kwargs,
        # device=device
    )
    output = model(input_tensor1)
    print("Model输入形状:", input_tensor1.shape)
    print("Model输出形状:", output[0].shape)
