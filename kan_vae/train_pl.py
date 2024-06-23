import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from kan_vae_model import KAN_VAE

# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# Define LightningModule
class VAEModule(pl.LightningModule):
    def __init__(self, input_dim, encoder_hidden_dims, decoder_hidden_dims, latent_dim):
        super(VAEModule, self).__init__()
        self.vae = KAN_VAE(
            input_dim, encoder_hidden_dims, decoder_hidden_dims, latent_dim
        )
        self.input_dim = input_dim

    def forward(self, x):
        return self.vae(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        reconstructed, mu, log_var = self.vae(x)
        loss = self.vae.loss_function(reconstructed, x, mu, log_var)
        self.log("train_loss", loss / len(x))
        return loss/len(x)

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        reconstructed, mu, log_var = self.vae(x)
        loss = self.vae.loss_function(reconstructed, x, mu, log_var)
        self.log("val_loss", loss / len(x))
        return loss/len(x)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.vae.parameters(), lr=1e-3)
        return optimizer


# Data preparation
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
trainset = torchvision.datasets.MNIST(
    root="/home/wangr/data/code/MyGithub/kan_vae/data",
    train=True,
    download=False,
    transform=transform,
)
valset = torchvision.datasets.MNIST(
    root="/home/wangr/data/code/MyGithub/kan_vae/data",
    train=False,
    download=False,
    transform=transform,
)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=False)

# Model initialization
input_dim = 28 * 28
encoder_hidden_dims = [256, 64]
decoder_hidden_dims = [64, 256]
latent_dim = 16

vae_module = VAEModule(input_dim, encoder_hidden_dims, decoder_hidden_dims, latent_dim)

# Checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="/home/wangr/data/code/MyGithub/kan_vae/model_save/",
    filename="kan_vae_mnist1d-{epoch:02d}-{val_loss:.2f}",
    save_top_k=3,
    mode="min",
)

# Trainer
trainer = Trainer(
    max_epochs=1000,
    accelerator="gpu",
    devices="-1",
    check_val_every_n_epoch=1,
    callbacks=[checkpoint_callback],
    log_every_n_steps=70,
    strategy="ddp_find_unused_parameters_true",
    # logger=logger,
)

# Training
trainer.fit(vae_module, trainloader, valloader)
