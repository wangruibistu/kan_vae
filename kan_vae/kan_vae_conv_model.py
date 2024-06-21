import torch
import torch.nn as nn
import torch.nn.functional as F
from kan_model import KAN_conv2d
from kan_conv_transpose2d import KAN_deconv2d
import math


class KAN_VAE2D(nn.Module):
    def __init__(
        self,
        input_channels,
        encoder_hidden_dims,
        decoder_hidden_dims,
        latent_dim,
        input_shape=(28, 28),
        encoderkwargs=None,
        decoderkwargs=None,
    ):
        super(KAN_VAE2D, self).__init__()
        self.encoder_hidden_dims = encoder_hidden_dims
        self.decoder_hidden_dims = decoder_hidden_dims
        encoderkwargs["layers_hidden"] = [input_channels] + encoder_hidden_dims
        decoderkwargs["layers_hidden"] = decoder_hidden_dims + [input_channels]
        decoderkwargs['input_shape'] = input_shape
        
        self.encoder = nn.Sequential(
            KAN_conv2d(encoderkwargs),
        )

        self.flatten = nn.Flatten()

        encoder_output_shape = calculate_encoder_output_shape(
            self.encoder, 
            (input_channels, *input_shape)
        )
        self.flattened_size = encoder_output_shape

        self.fc_mu = nn.Linear(
            encoder_output_shape[0] * encoder_output_shape[1] * encoder_output_shape[2],
            latent_dim,
        )

        self.fc_var = nn.Linear(
            encoder_output_shape[0] * encoder_output_shape[1] * encoder_output_shape[2],
            latent_dim,
        )

        # self.fc_mu = nn.Conv2d(encoder_hidden_dims[-1], latent_dim, kernel_size=1)
        # self.fc_var = nn.Conv2d(encoder_hidden_dims[-1], latent_dim, kernel_size=1)

        self.decoder_input = nn.Linear(
            latent_dim,
            encoder_output_shape[0] * encoder_output_shape[1] * encoder_output_shape[2],
        )

        self.decoder = nn.Sequential(
            KAN_deconv2d(decoderkwargs),
            nn.GELU(),
        )

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
            self.flattened_size[0],
            self.flattened_size[1],
            self.flattened_size[2],
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


def calculate_encoder_output_shape(encoder, input_shape):
    with torch.no_grad():
        dummy_input = torch.zeros(1, *input_shape)
        output = encoder(dummy_input)
        return output.shape[1:]

