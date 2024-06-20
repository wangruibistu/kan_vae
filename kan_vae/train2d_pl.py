import os
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from kan_vae_model import KAN_VAE2D

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")


class KANVAE2DLightning(LightningModule):
    def __init__(self):
        super(KANVAE2DLightning, self).__init__()
        self.model = KAN_VAE2D(
            input_channels=1,
            encoder_hidden_dims=[8, 16, 32],
            decoder_hidden_dims=[32, 16, 8],
            latent_dim=64,
            input_shape=(28, 28),
            encoderkwargs={
                "kernel_sizes": [3, 3, 3],
                "strides": [2, 2, 2],
                "paddings": [1, 1, 1],
                "grid_size": 5,
                "spline_order": 3,
                "scale_noise": 0.1,
                "scale_base": 1.0,
                "scale_spline": 1.0,
                "base_activation": torch.nn.SiLU,
                "grid_eps": 0.02,
                "grid_range": [-1, 1],
            },
            decoderkwargs={
                "kernel_sizes": [3, 3, 3],
                "strides": [2, 2, 2],
                "paddings": [1, 1, 1],
                "output_paddings": [0, 1, 1],
                "grid_size": 5,
                "spline_order": 3,
                "scale_noise": 0.1,
                "scale_base": 1.0,
                "scale_spline": 1.0,
                "base_activation": torch.nn.SiLU,
                "grid_eps": 0.02,
                "grid_range": [-1, 1],
            },
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        reconstructed, mu, log_var = self.model(x)
        loss = self.model.loss_function(reconstructed, x, mu, log_var)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        reconstructed, mu, log_var = self.model(x)
        loss = self.model.loss_function(reconstructed, x, mu, log_var)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=1e-3)


def model_train():
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

    trainloader = DataLoader(trainset, batch_size=16, shuffle=True)
    valloader = DataLoader(valset, batch_size=16, shuffle=False)

    model = KANVAE2DLightning()

    logger = TensorBoardLogger("tb_logs", name="kan_vae_mnist")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="/home/wangr/data/code/MyGithub/kan_vae/model_save/",
        filename="kan_vae_mnist2d-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    trainer = Trainer(
        max_epochs=100,
        accelerator="gpu",
        devices="-1",
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback],
        log_every_n_steps=70,
        strategy="ddp_find_unused_parameters_true",
        logger=logger
    )
    trainer.fit(
        model=model, 
        train_dataloaders=trainloader, 
        val_dataloaders=valloader
    )


if __name__ == "__main__":
    model_train()
