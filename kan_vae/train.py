import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Train on MNIST
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from kan_vae_model import KAN_VAE
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

input_dim = 28 * 28 
encoder_hidden_dims = [256, 64]
decoder_hidden_dims = [64, 256]
latent_dim = 16

vae = KAN_VAE(
    input_dim,
    encoder_hidden_dims,
    decoder_hidden_dims,
    latent_dim,
).to(device)

optimizer = optim.Adam(vae.parameters(), lr=1e-3)

num_epochs = 1000
for epoch in range(num_epochs):
    vae.train()
    train_loss = 0
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}")

    for batch in progress_bar:
        x, _ = batch  
        x = x.view(x.size(0), -1).to(device)

        optimizer.zero_grad()
        reconstructed, mu, log_var = vae(x)
        loss = vae.loss_function(reconstructed, x, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        progress_bar.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1}, Loss: {train_loss/len(trainloader.dataset)}")


torch.save(
    vae.state_dict(), 
    "/home/wangr/data/code/MyGithub/kan_vae/model_save/kan_vae_mnist1d.pth"
)
