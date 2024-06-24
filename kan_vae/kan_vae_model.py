import torch
import torch.nn as nn
import torch.nn.functional as F
from kan_linear import KAN  # KAN_deconv2d

import math


class KAN_VAE(nn.Module):
    def __init__(
        self, input_dim, encoder_hidden_dims, decoder_hidden_dims, latent_dim, **kwargs
    ):
        super(KAN_VAE, self).__init__()
        self.encoder = KAN(layers_hidden=[input_dim] + encoder_hidden_dims)

        self.fc_mu = nn.Linear(encoder_hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(encoder_hidden_dims[-1], latent_dim)

        self.decoder = KAN(layers_hidden=[latent_dim] + decoder_hidden_dims)
        self.fc_output = nn.Linear(decoder_hidden_dims[-1], input_dim)

    def encode(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        log_var = self.fc_var(encoded)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        decoded = self.decoder(z)
        return self.fc_output(decoded)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode(z)
        return reconstructed, mu, log_var

    def loss_function(self, reconstructed, x, mu, log_var):
        recon_loss = F.mse_loss(reconstructed, x, reduction="sum")
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + kld_loss
