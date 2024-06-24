KAN-VAE Project

Overview

This repository contains the implementation of a Variational Autoencoder (VAE) using KAN convolutions for encoding and decoding. The KAN-VAE model is designed for image reconstruction tasks, utilizing custom convolutional layers to enhance the learning process.

Features

	• Custom KAN convolutional and transposed convolutional layers.
	• Integration with PyTorch Lightning for easy training and validation.
	• Configurable hyperparameters for the KAN layers and VAE architecture.
	• Example training script for the MNIST dataset.

Requirements

	• Python 3.7+
	• PyTorch 1.8+
	• PyTorch Lightning 1.3+
	• torchvision 0.9+
	• tqdm

Installation

    1. Clone the repository:
       git clone https://github.com/yourusername/kan-vae.git
       cd kan-vae
    2. Install the required packages:
       pip install -r requirements.txt
       make install

Usage

Training the Model

To train the KAN-VAE model on the MNIST dataset, you can use the provided training script:

	1.  Ensure you have the MNIST dataset downloaded in the specified directory or modify the paths as needed.
	2.  Run the training script:
        python train2d_pl.py

Configuration

You can modify the configuration parameters for the KAN layers and VAE model in the train.py/train2d.py file.

Acknowledgements

This project was made possible with the support and inspiration from the following works:

	• https://github.com/KindXiaoming/pykan
	• https://github.com/Blealtan/efficient-kan
	• https://github.com/AntonioTepsich/Convolutional-KANs
	• https://github.com/StarostinV/convkan

License

This project is licensed under the MIT License. See the LICENSE file for details.

Contact

For any inquiries or issues, please contact wangrui@nao.cas.cn

