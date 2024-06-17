import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from kan_vae_model import KAN_VAE

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载 MNIST 数据集
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

# 定义模型和优化器
input_dim = 28 * 28  # MNIST 输入维度
hidden_dims = [512, 256]
latent_dim = 64

vae = KAN_VAE(
    input_dim,
    hidden_dims,
    latent_dim,
).to(
    device
)  # 将模型移动到GPU

optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    vae.train()
    train_loss = 0
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}")

    for batch in progress_bar:
        x, _ = batch
        x = x.view(x.size(0), -1).to(device)  # 将数据移动到GPU

        optimizer.zero_grad()
        reconstructed, mu, log_var = vae(x)
        loss = vae.loss_function(reconstructed, x, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        progress_bar.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1}, Loss: {train_loss/len(trainloader.dataset)}")

    # 验证模型
    vae.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in valloader:
            x, _ = batch
            x = x.view(x.size(0), -1).to(device)

            reconstructed, mu, log_var = vae(x)
            loss = vae.loss_function(reconstructed, x, mu, log_var)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}, Validation Loss: {val_loss/len(valloader.dataset)}")

# 保存模型
torch.save(
    vae.state_dict(),
    "/home/wangr/data/code/MyGithub/kan_vae/model_save/kan_vae_mnist.pth",
)
