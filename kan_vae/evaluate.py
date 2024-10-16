import os
import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from kan_vae_model import KAN_VAE
from kan_vae_2d_model import KAN_VAE_model


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose(
    [
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
    ]
)

valset = torchvision.datasets.MNIST(
    root="/home/wangr/code/efficient-kan/src/data",
    train=False,
    download=False,
    transform=transform,
)

valloader = DataLoader(
    valset, batch_size=64, shuffle=False
)

input_dim = 28 * 28
encoder_hidden_dims = [256, 64]
decoder_hidden_dims = [64, 256]
latent_dim = 16

# model = KAN_VAE(
#     input_dim,
#     encoder_hidden_dims,
#     decoder_hidden_dims,
#     latent_dim,
# ).to(device)

model = KAN_VAE_model(
    input_channels=1,
    input_shape=(28, 28),
    latent_dim=16,
    encoderkwargs={
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
        "device": device,
    },
    decoderkwargs={
        "layers_hidden": [32, 16, 1],
        "kernel_sizes": [(3, 3), (3, 3)],
        "strides": [(2, 2), (2, 2)],
        "paddings": [(1, 1), (1, 1)],
        "output_paddings": [(1, 1), (1, 1)],
        "groups": 1,
        "grid_size": 5,
        "spline_order": 3,
        "dilation": (1, 1),
        "device": device,
    },
)

# model_path = "/home/wangr/data/code/MyGithub/kan_vae/model_save/kan_vae_mnist.pth"
model_path = "/home/wangr/data/code/MyGithub/kan_vae/model_save/kan_vae_mnist2d-epoch=208-val_loss=64.61.pth"

model.load_state_dict(torch.load(model_path))
model.eval()

val_loss = 0
reconstruction_errors = []
originals = []
reconstructions = []

with torch.no_grad():
    for batch in valloader:
        x, _ = batch
        x = x.view(x.size(0), -1).to(device)

        reconstructed, mu, log_var = model(x)
        loss = model.loss_function(reconstructed, x, mu, log_var)
        val_loss += loss.item()

        reconstruction_error = torch.mean((reconstructed - x) ** 2, dim=1)
        reconstruction_errors.extend(reconstruction_error.cpu().numpy())

        originals.append(x.cpu().view(-1, 28, 28))
        reconstructions.append(reconstructed.cpu().view(-1, 28, 28))

average_val_loss = val_loss / len(valloader.dataset)
print(f"Validation Loss: {average_val_loss}")

reconstruction_errors = torch.tensor(reconstruction_errors)
print(
    f"Reconstruction Error - Mean: {reconstruction_errors.mean().item()}, Std: {reconstruction_errors.std().item()}"
)


def show_images(originals, reconstructions, n=10):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(originals[i], cmap="gray")
        ax.axis("off")

        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(reconstructions[i], cmap="gray")
        ax.axis("off")

        ax = plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(originals[i] - reconstructions[i], cmap="gray")
        ax.axis("off")


originals = torch.cat(originals)[:10]
reconstructions = torch.cat(reconstructions)[:10]
show_images(originals, reconstructions, n=10)
plt.show()
