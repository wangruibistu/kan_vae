import os
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from kan_vae_model import KAN_VAE2D 


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

trainset = torchvision.datasets.MNIST(
    root="/home/wangr/code/efficient-kan/src/data",
    train=True,
    download=False,
    transform=transform,
)

valset = torchvision.datasets.MNIST(
    root="/home/wangr/code/efficient-kan/src/data",
    train=False,
    download=False,
    transform=transform,
)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=False)

input_channels = 1  
encoder_hidden_dims = [32, 64]
decoder_hidden_dims = [64, 32]
latent_dim = 64

model = KAN_VAE2D(
    input_channels,
    encoder_hidden_dims,
    decoder_hidden_dims,
    latent_dim,
    kernel_size=3,
    stride=2,
    padding=1,
    grid_size=5,
    spline_order=3,
    scale_noise=0.1,
    scale_base=1.0,
    scale_spline=1.0,
    base_activation=torch.nn.SiLU,
    grid_eps=0.02,
    grid_range=[-1, 1],
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}")

    for batch in progress_bar:
        x, _ = batch
        x = x.to(device)  

        optimizer.zero_grad()
        reconstructed, mu, log_var = model(x)
        loss = model.loss_function(reconstructed, x, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        progress_bar.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1}, Loss: {train_loss/len(trainloader.dataset)}")

    model.eval()
    val_loss = 0
    reconstruction_errors = []
    originals = []
    reconstructions = []

    with torch.no_grad():
        for batch in valloader:
            x, _ = batch
            x = x.to(device)

            reconstructed, mu, log_var = model(x)
            loss = model.loss_function(reconstructed, x, mu, log_var)
            val_loss += loss.item()

            reconstruction_error = torch.mean((reconstructed - x) ** 2, dim=[1, 2, 3])
            reconstruction_errors.extend(reconstruction_error.cpu().numpy())

            originals.append(x.cpu())
            reconstructions.append(reconstructed.cpu())

    average_val_loss = val_loss / len(valloader.dataset)
    print(f"Epoch {epoch+1}, Validation Loss: {average_val_loss}")

torch.save(
    model.state_dict(),
    "/home/wangr/data/code/MyGithub/kan_vae/model_save/kan_vae_mnist2d.pth",
)

import matplotlib.pyplot as plt


def show_images(originals, reconstructions, n=10):
    plt.figure(figsize=(20, 6))
    for i in range(n):
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(originals[i][0], cmap="gray")
        ax.axis("off")

        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(reconstructions[i][0], cmap="gray")
        ax.axis("off")

        ax = plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow((originals[i] - reconstructions[i])[0], cmap="gray")
        ax.axis("off")

originals = torch.cat(originals)[:10]
reconstructions = torch.cat(reconstructions)[:10]
show_images(originals, reconstructions, n=10)
plt.show()
