# no sigmoid
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable

from distutils import util
import numpy as np
import sys

sys.path.append('/Users/wangr/code/')
import collections

sys.path.append('/Users/wangr/code/efficient-kan/src/')
from efficient_kan import KAN


dtype = torch.cuda.FloatTensor
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def init_weights(m):
    """
    Glorot uniform initialization for network.
    """
    if 'conv' in m.__class__.__name__.lower():
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def compute_out_size(in_size, mod):
    """
    Compute output size of Module `mod` given an input with size `in_size`.
    """

    f = mod.forward(autograd.Variable(torch.Tensor(1, *in_size)))
    return f.size()[1:]


def same_padding(input_pixels, filter_len, stride=1):
    effective_filter_size_rows = (filter_len - 1) + 1
    output_pixels = (input_pixels + stride - 1) // stride
    padding_needed = max(0, (output_pixels - 1) * stride + effective_filter_size_rows - input_pixels)
    padding = max(0, (output_pixels - 1) * stride + (filter_len - 1) + 1 - input_pixels)
    rows_odd = (padding % 2 != 0)
    return padding // 2


def build_encoder(input_filts, conv_filts, conv_strides, conv_filt_lens, activation, out_norm=True, z_filts=25,
                  output_z=True, init=True, use_cuda=True):
    layers = []

    # Conv layers
    for filts, strides, filter_length in zip(conv_filts, conv_strides, conv_filt_lens):
        layers.append(nn.Conv1d(input_filts, filts, filter_length, strides))
        layers.append(activation)
        input_filts = filts

    if output_z:
        # Latent output
        layers.append(nn.Conv1d(input_filts, z_filts, 1, 1))
        if out_norm:
            layers.append(nn.InstanceNorm1d(z_filts))
    model = torch.nn.Sequential(*layers)

    if init:
        # Initialize weights and biases
        model.apply(init_weights)

    if use_cuda:
        # Switch to GPU
        model = model.cuda()

    return model


def build_decoder(input_filts, conv_filts, conv_strides, conv_filt_lens, activation, output_x=True, init=True,
                  use_cuda=True):
    layers = []

    # Conv layers (reverse order of encoders)
    for filts, strides, filter_length in zip(reversed(conv_filts), reversed(conv_strides), reversed(conv_filt_lens)):
        # if strides>1:
        layers.append(nn.ConvTranspose1d(input_filts, filts, filter_length, strides, output_padding=1))
        # else:
        #    layers.append(nn.Conv1d(input_filts, filts, filter_length, strides))
        layers.append(activation)
        input_filts = filts

    if output_x == True:
        # Spectrum output
        layers.append(nn.Conv1d(in_channels=input_filts, out_channels=1, kernel_size=53, stride=1))
    model = torch.nn.Sequential(*layers)

    if init:
        # Initialize weights and biases
        model.apply(init_weights)

    if use_cuda:
        model = model.cuda()

    return model


class KAN_VAE(nn.Module):

    def __init__(self, architecture_config, use_cuda=True):
        super(KAN_VAE, self).__init__()

        # Read configuration
        num_pixels = int(architecture_config['num_pixels'])
        activation = architecture_config['activation']
        conv_filts_ae_dom = eval(architecture_config['conv_filts_ae_dom'])
        conv_filt_lens_ae_dom = eval(architecture_config['conv_filt_lens_ae_dom'])
        conv_strides_ae_dom = eval(architecture_config['conv_strides_ae_dom'])
        conv_filts_ae_sh = eval(architecture_config['conv_filts_ae_sh'])
        conv_filt_lens_ae_sh = eval(architecture_config['conv_filt_lens_ae_sh'])
        conv_strides_ae_sh = eval(architecture_config['conv_strides_ae_sh'])
        shared_z_filters = int(architecture_config['shared_z_filters'])
        split_z_filters = int(architecture_config['split_z_filters'])
        # This variable is used for spectra fitting only:
        self.cur_z_sp = None
        self.x_mean = None
        self.x_std = None

        # Whether or not to use a split latent-space
        if split_z_filters > 0:
            self.use_split = True
        else:
            self.use_split = False

        # Define activation function
        if activation.lower() == 'sigmoid':
            activ_fn = torch.nn.Sigmoid()
        elif activation.lower() == 'leakyrelu':
            activ_fn = torch.nn.LeakyReLU(0.1)
        elif activation.lower() == 'relu':
            activ_fn = torch.nn.ReLU()

        # Build encoding networks
        self.encoder= build_encoder(
            1, 
            conv_filts_ae_dom, 
            conv_strides_ae_dom,
            conv_filt_lens_ae_dom, 
            activ_fn,
            output_z=False, 
            init=True,
            use_cuda=use_cuda
            )
       # Build decoding networks
        # print(conv_filts_ae_sh,conv_strides_ae_sh,conv_filt_lens_ae_sh)
        self.decoder = build_decoder(
            shared_z_filters, 
            conv_filts_ae_sh, 
            conv_strides_ae_sh, 
            conv_filt_lens_ae_sh,
            activ_fn, 
            output_x=False, 
            init=True, 
            use_cuda=use_cuda
            )

        # Infer output shapes of each model
        self.enc_interm_shape = compute_out_size((1, num_pixels), self.encoder_synth)
        self.z_sh_shape = compute_out_size(self.enc_interm_shape, self.encoder_sh)
        if self.use_split:
            self.z_sp_shape = compute_out_size(self.enc_interm_shape, self.encoder_sp)
        self.dec_interm_shape = compute_out_size(self.z_sh_shape, self.decoder_sh)

    def y_to_synth(self, y, use_cuda=True):
        if use_cuda:
            y = y.cuda()
            y_min = self.y_min.cuda()
            y_max = self.y_max.cuda()
        else:
            y = y.cpu()
            y_min = self.y_min.cpu()
            y_max = self.y_max.cpu()

        y = (y - y_min) / (y_max - y_min) - 0.5
        return self.emulator(y)


    def obs_to_z(self, x):
        interm = self.encoder_obs(x.unsqueeze(1))
        if self.use_split:
            return self.encoder_sh(interm), self.encoder_sp(interm)
        else:
            return self.encoder_sh(interm)

    def z_to_obs(self, z_sh, z_sp=None):
        if self.use_split:
            # print(z_sh.shape,z_sp.shape)
            return self.decoder_obs(torch.cat((self.decoder_sh(z_sh), self.decoder_sp(z_sp)), 1)).squeeze(1)
        else:
            return self.decoder_obs(self.decoder_sh(z_sh)).squeeze(1)

    def dis_train_mode(self):
        self.encoder.train()
        self.decoder.train()

    def eval_mode(self):
        self.encoder.eval()
        self.decoder.eval()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
