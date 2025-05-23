import av
import sounddevice as sd

import itertools

import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

import torch
from torch import nn

import torchaudio
from torchaudio.transforms import MelSpectrogram, Spectrogram, AmplitudeToDB, InverseMelScale, GriffinLim, InverseSpectrogram

from utils import *

import random

import pandas as pd

import gc
import time
from importlib import reload
import os
import inspect

KERNEL_SIZE2D=(3,4)
#PADDING=0
STRIDE2D=(2,2)
BINS=241
def auto_save_hyperparams(init_fn):
    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
    def get_config(self):
        return self.hparams
        
    def wrapper(self, *args, **kwargs):
        # Bind the arguments to the function signature and apply defaults
        sig = inspect.signature(init_fn)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()
        # Save all parameters except 'self'
        self.hparams = {
            name: value 
            for name, value in bound_args.arguments.items() 
            if name != "self"
        }
        self.from_config = from_config
        self.get_config = lambda: get_config(self)
        return init_fn(self, *args, **kwargs)
    return wrapper


class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = 0.0,
        stop: float = 1.0,
        num_gaussians: int = 24,
    ):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

class MLP(nn.Module):
    @auto_save_hyperparams
    def __init__(self, 
                input_dim, 
                hidden_dims, 
                output_dim, 
                norm=nn.LayerNorm, 
                final_norm=nn.Identity, 
                activation=nn.PReLU, 
                final_activation=nn.Identity, 
                dropout_rate=0.1, 
                final_dropout_rate=0.0,
                ):
        super().__init__()
        dims=[input_dim]+hidden_dims+[output_dim]
        self.lins=nn.ModuleList()
        self.norms=nn.ModuleList()
        self.acts=nn.ModuleList()
        self.dropouts=nn.ModuleList()
        for i in range(len(dims)-1):
            self.lins.append(nn.Linear(dims[i], dims[i+1]))
            if i+1<len(dims)-1:
                self.norms.append(norm(dims[i+1]))
                self.acts.append(activation())
                self.dropouts.append(nn.Dropout(dropout_rate))
            else:
                self.norms.append(final_norm(dims[i+1]))
                self.acts.append(final_activation())
                self.dropouts.append(nn.Dropout(final_dropout_rate))
        self.reset_parameters()
    def reset_parameters(self):
        for layer in self.lins:
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0)
    def forward(self, x):
        for lin, norm, act, do in zip(self.lins, self.norms, self.acts, self.dropouts):
            x=lin(x)
            x=norm(x)
            x=act(x)
            x=do(x)
        return x
        
class UNet2d4(nn.Module):
    @auto_save_hyperparams
    def __init__(self, chnls_in=1, chnls_out=1, chnls_gs=32, dropout=0.01):
        super().__init__()
        I=chnls_in
        A=64
        B=64
        C=64
        D=64
        E=64
        F=64
        G=64
        O=chnls_out
        S=chnls_gs
        self.gs = GaussianSmearing(num_gaussians=S)
        
        #self.batch_norm = nn.BatchNorm2d(chnls_in)
        self.dcl_1 = DownConvBlock2d(I+S, A, kernel_size=3, stride=2,                       dropout=dropout)
        self.dcl_2 = DownConvBlock2d(A,   B, kernel_size=3, stride=2,                       dropout=dropout)
        self.dcl_3 = DownConvBlock2d(B,   C, kernel_size=3, stride=2,                       dropout=dropout)
        self.dcl_4 = DownConvBlock2d(C,   D, kernel_size=3, stride=2,                       dropout=dropout)
        self.dcl_5 = DownConvBlock2d(D,   E, kernel_size=3, stride=2,         norm=False,   dropout=dropout)
        self.dcl_6 = DownConvBlock2d(E,   F, kernel_size=(4,3), stride=(3,2), norm=False,   dropout=dropout)
        
        self.ucl_1 = UpConvBlock2d  (F,   E, kernel_size=(4,3), stride=(3,2), output_padding=0,     dropout=dropout)
        self.ucl_2 = UpConvBlock2d  (E+E, D, kernel_size=3,     stride=2,     output_padding=(1,1), dropout=dropout)
        self.ucl_3 = UpConvBlock2d  (D+D, C, kernel_size=3,     stride=2,     output_padding=(0,1), dropout=dropout)
        self.ucl_4 = UpConvBlock2d  (C+C, B, kernel_size=3,     stride=2,     output_padding=0,     dropout=dropout)
        self.ucl_5 = UpConvBlock2d  (B+B, A, kernel_size=3,     stride=2,     output_padding=(0,1), dropout=dropout)
        self.ucl_0 =nn.ConvTranspose2d(A+A,O,kernel_size=3,     stride=2,     output_padding=0,     padding=1)
    
    def forward(self, logmag):
        #logmag = spec.abs().log()
        
        n_channels = logmag.shape[-3]
        n_frames = logmag.shape[-1]
        logmag=logmag.unsqueeze(-3)
        #x=self.transform(x)
        #if len(x.shape)==3:
        #    x=x.unsqueeze(0)
        #x=self.batch_norm(x)
        ####print("doing dcl_1", x.shape)
        gs=self.gs(torch.linspace(0,1,BINS).to(self.gs.offset.device).sqrt()).broadcast_to(n_channels,n_frames,BINS,-1).permute(0,-1,2,1)
        #print("logmag:",logmag.shape)
        #print("phase:",phase.shape)
        #print("gs:",gs.shape)
        #gs=unwrap_complex(gs)
        #print("x:",x.shape)
        #print("gs:",gs.shape)
        enc1 = self.dcl_1(torch.cat([logmag,gs],-3))
        #print("doing dcl_2", enc1.shape)
        enc2 = self.dcl_2(enc1) 
        #print("doing dcl_3", enc2.shape)
        enc3 = self.dcl_3(enc2)
        #print("doing dcl_4", enc3.shape)
        enc4 = self.dcl_4(enc3)
        #print("doing dcl_5", enc4.shape)
        enc5 = self.dcl_5(enc4)
        #print("doing dcl_6", enc5.shape)
        enc6 = self.dcl_6(enc5)
        ##print("doing dcl_7", enc6.shape)
        #enc7 = self.dcl_7(enc6)
 
        #print("doing ucl_1", enc6.shape, enc6.shape)
        dec1 = self.ucl_1(enc6, enc5)
        #print("doing ucl_2", dec1.shape, enc4.shape)
        dec2 = self.ucl_2(dec1, enc4)
        #print("doing ucl_3", dec2.shape, enc3.shape)
        dec3 = self.ucl_3(dec2, enc3)
        #print("doing ucl_4", dec3.shape, enc2.shape)
        dec4 = self.ucl_4(dec3, enc2)
        #print("doing ucl_5", dec4.shape, enc1.shape)
        dec5 = self.ucl_5(dec4, enc1)
        ##print("doing ucl_6", dec5.shape, enc1.shape)
        #dec6 = self.ucl_6(dec5, enc1)
        #print("doing ucl_0", dec5.shape)
        dec0 = self.ucl_0(dec5)
        final=dec0.squeeze(-3)
        return final
        #final = torch.polar(final.exp(), phase)
        ####print("doing upsample_layer", dec5.shape)
        #final = self.upsample_layer(dec5)
        ####print("doing zero_pad", dec5.shape)
        #final = self.zero_pad(dec5)
        ####print("doing ucl_7", dec6.shape)
        #print("final transformations",dec0.shape)
        #final = self.activation(self.mlp(self.pre_mlp_act(dec0).transpose(-1,-2)).transpose(-1,-2))
        #final = self.inverse_transform(final)
        ####print("returning",final.shape)
        
    #def transform(self, spectrogram):
    #    return spectrogram.flatten(-3,-2)
    #def inverse_transform(self, spectrogram):
    #    return spectrogram.unflatten(-2,(2,-1))

class UpConvBlock2d(nn.Module):
    @auto_save_hyperparams
    def __init__(self, ip_sz, op_sz, kernel_size=KERNEL_SIZE2D, stride=STRIDE2D, padding=1, output_padding=0 ,dropout=None, norm = True):
        super().__init__()
        self.layers = [
            nn.ConvTranspose2d(ip_sz, op_sz, kernel_size=kernel_size, padding=padding, stride=stride, output_padding=output_padding),
        ]
        if norm:
            self.layers += [nn.InstanceNorm2d(op_sz)]
        self.layers += [nn.PReLU()]
        if dropout:
            self.layers += [nn.Dropout(dropout)]
        self.layers=nn.Sequential(*(self.layers))
    def forward(self, x, enc_ip):
        ##print("    doing sequentilal", x.shape)
        x = self.layers(x)
        ##print("    doing cat", x.shape, enc_ip.shape)
        op = torch.cat((x, enc_ip), -3)
        ##print("    returning",op.shape)
        return op


class DownConvBlock2d(nn.Module):
    @auto_save_hyperparams
    def __init__(self, ip_sz, op_sz, kernel_size=KERNEL_SIZE2D, stride=STRIDE2D, padding=1, norm=True, dropout=None):
        super().__init__()
        self.layers = [nn.Conv2d(ip_sz, op_sz, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm is True:
            self.layers.append(nn.InstanceNorm2d(op_sz))
        elif isinstance(norm, tuple):
            self.layers.append(nn.LayerNorm(norm))
        self.layers += [nn.PReLU()]
        if dropout:
            self.layers += [nn.Dropout(dropout)]
        self.layers=nn.Sequential(*(self.layers))
    def forward(self, x):
        op = self.layers(x)#[...,:-1]
        return op


