import torch.nn.functional as F
import os
import os.path
import random
from pathlib import Path
import math
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
import torch
import pypianoroll
from pypianoroll import Multitrack, Track
from tqdm.notebook import tqdm
import time
nfea = 16
latent_v_dim = 32
n_fc = 6
n_verses = 6
latent_w = 4
latent_h = 6
batch_size = 32
latent_dim = 8

n_measures = 4  # number of measures per sample
beat_resolution = 4
n_pitches = 72  # number of pitches
n_measures = 4  # number of measures per sample
measure_resolution = 4 * beat_resolution
beat_resolution = 4  # temporal resolution of a beat (in timestep)

embedding_dim = latent_dim
num_embeddings = 2048
commitment_cost = 0.25
decay = 0.99
class MaskedConv3d(torch.nn.Conv3d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv3d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kI, kP, kT = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, :, :, kT - 1 + (mask_type == 'B'):] = 0
    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv3d, self).forward(x)
class MaskBlock(torch.nn.Module):
    def __init__(self, mask_type, in_dim, out_dim, kernel):
        super().__init__()
        self.n_tracks = 5
        self.n_classes = 6
        self.kernel = kernel
        self.out_dim = out_dim
        self.padding = kernel - 1
        self.conv0 = torch.nn.Conv3d(in_dim , out_dim * self.n_classes, (1, self.n_classes, 1), 1, 0, bias=False)
        self.conv1 = torch.nn.Conv3d(out_dim, out_dim *  self.n_tracks, (1, self.n_tracks, 1),       1, 0, bias=False)
        self.maskconv = MaskedConv3d(mask_type ,
                                     out_dim, out_dim                 , (1, 1, kernel),  1, (0, 0, self.padding), bias=False)
        self.batchnorm = torch.nn.BatchNorm3d(out_dim)
        self.relu = torch.nn.ReLU(True)
    def forward(self, x):
        x_shape = x.shape
        x = self.conv0(x)
        x = x.view(x_shape[0], self.out_dim, self.n_classes , x_shape[2], x_shape[4])
        x = self.conv1(x)
        x = x.view(x_shape[0], self.out_dim, self.n_tracks , self.n_classes, x_shape[4])
        x = self.maskconv(x)
        x = x[:, :, :, :, : -(self.kernel - 1)].view(x_shape[0], self.out_dim, x_shape[2], x_shape[3], x_shape[4])
        x = self.batchnorm(x)
        x = self.relu(x)
        return x
    
class PIXELCNN(torch.nn.Module):
    # 1 chanel PixelCNN
    def __init__(self, k_dim, z_dim, kernel_size=3, fm=64):
        super(PIXELCNN, self).__init__()
        self.k_dim = k_dim
        self.z_dim = z_dim
        self.fm = fm
        self.kernel_size = kernel_size
        self.padding = kernel_size
        self.latent_p = 6
        self.conv0 = MaskBlock('A', self.z_dim, self.fm, kernel_size)
        self.num_b_conv = 7
        self.conv1 = torch.nn.ModuleList([
            torch.nn.Sequential(
                MaskBlock('B', self.fm, self.fm, kernel_size),
            ) for _ in range(self.num_b_conv)
        ])
        self.conv2 = torch.nn.Conv3d(self.fm, self.k_dim, 1, 1, 0)
        

    def forward(self, z):
        x_shape = z.shape
        x = self.conv0(z)
        for i in range(self.num_b_conv):
            x = self.conv1[i](x)
        x = self.conv2(x)
        return x


    
class GeneraterBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride):
        super().__init__()
        self.transconv = torch.nn.ConvTranspose3d(in_dim, out_dim, kernel, stride)
        self.batchnorm = torch.nn.BatchNorm3d(out_dim)
    
    def forward(self, x):
        x = self.transconv(x)
        x = self.batchnorm(x)
        return torch.nn.functional.relu(x)


class Bar_Decoder(torch.nn.Module):
    """A convolutional neural network (CNN) based generator. The generator takes
    as input a latent vector and outputs a fake sample."""
    def __init__(self):
        super().__init__()
        # 
        self.transconv0 = GeneraterBlock(latent_dim, nfea * 8, (1, 1, 1), (1, 1, 1))
        self.transconv1 = GeneraterBlock(nfea * 8, nfea *  4, (1, 4, 1), (1, 4, 1))
        self.transconv2 = GeneraterBlock(nfea *  4, nfea *  2, (1, 1, 3), (1, 1, 3))
        self.transconv3 = GeneraterBlock(nfea *  2, nfea *  1, (1, 4, 1), (1, 4, 1))
        self.transconv4 = torch.nn.ConvTranspose3d(nfea *  1,         1, (1, 1, 4), (1, 1, 4))
        layers = []
        for i in range(4):
            layers.append(torch.nn.Linear(latent_dim, latent_dim))
            layers.append(torch.nn.LeakyReLU(0.2))
        self.mapping = torch.nn.Sequential(*layers)
    def forward(self, x):
        #merge latent
        x = x.view(-1, latent_dim, latent_w, 1, latent_h)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x_shape = x.shape
        flat_x = x.view(-1, latent_dim)
        flat_x = self.mapping(flat_x)
        x = flat_x.view(x_shape)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        
        
        x = self.transconv0(x)
        x = self.transconv1(x)
        x = self.transconv2(x)
        x = self.transconv3(x)
        x = self.transconv4(x)
        x = x.view(-1,  n_measures * measure_resolution, n_pitches)
        return x

class LayerNorm(torch.nn.Module):
    """An implementation of Layer normalization that does not require size
    information. Copied from https://github.com/pytorch/pytorch/issues/1959."""
    def __init__(self, n_features, eps=1e-5, affine=True):
        super().__init__()
        self.n_features = n_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = torch.nn.Parameter(torch.Tensor(n_features).uniform_())
            self.beta = torch.nn.Parameter(torch.zeros(n_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y
class CompressBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride):
        super().__init__()
        self.transconv = torch.nn.Conv3d(in_dim, out_dim, kernel, stride)
        self.layernorm = LayerNorm(out_dim)
    
    def forward(self, x):
        x = self.transconv(x)
        x = self.layernorm(x)
        return torch.nn.functional.leaky_relu(x)
class Bar_Encoder(torch.nn.Module):
    """A convolutional neural network (CNN) based discriminator. The
    discriminator takes as input either a real sample (in the training data) or
    a fake sample (generated by the generator) and outputs a scalar indicating
    its authentity.
    """
    def __init__(self):
        super().__init__()
        self.conv0 = CompressBlock(        1, nfea *  1, (1, 1, 4), (1, 1, 4)) 
        self.conv1 = CompressBlock(nfea *  1, nfea *  2, (1, 4, 1), (1, 4, 1))
        self.conv2 = CompressBlock(nfea *  2, nfea *  4, (1, 1, 3), (1, 1, 3))
        self.conv3 = CompressBlock(nfea *  4, nfea * 8, (1, 4, 1), (1, 4, 1))
        self.conv4 = torch.nn.Conv3d(nfea * 8, latent_dim, (1, 1, 1), (1, 1, 1))
        layers = []
        for i in range(3):
            layers.append(torch.nn.Linear(latent_dim, latent_dim))
            layers.append(torch.nn.LeakyReLU(0.2))
        layers.append(torch.nn.Linear(latent_dim, latent_dim))
        self.mapping = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        x = x.view(-1, 1, n_measures, measure_resolution, n_pitches)

        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)         
        x = self.conv4(x)

        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x_shape = x.shape
        flat_x = x.view(-1, latent_dim)
        flat_x = self.mapping(flat_x)
        x = flat_x.view(x_shape)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = x.view(-1, latent_dim, latent_w, latent_h)
        return x

class VectorQuantizer(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = torch.nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cosmeasure_resolutiont = commitment_cost

    def forward(self, inputs):
        
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t()))    
        # Encoding
        encoding_indices = torch.argminDecoder(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
class VectorQuantizerEMA(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = torch.nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = torch.nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = torch.nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = torch.nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        encodings = encodings.view(input_shape[0], input_shape[1], input_shape[2], self._num_embeddings).permute(0, 3, 1, 2).contiguous()
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
        
class Model(torch.nn.Module):
    def __init__(self, commitment_cost, decay=0):
        super(Model, self).__init__()
        
        self.encoder = Bar_Encoder()
        if decay > 0.0:
            self.VQ = VectorQuantizerEMA(num_embeddings, embedding_dim, 
                                              commitment_cost, decay)

        else:
            self.VQ = VectorQuantizer(num_embeddings, embedding_dim, 
                                              commitment_cost, decay)
            
        self.decoder = Bar_Decoder()

    def forward(self, x):
        z = self.encoder(x)
        loss, z_q, perplexity, _ = self.VQ(z)
        x_recon = self.decoder(z_q)
        return loss, x_recon, perplexity


