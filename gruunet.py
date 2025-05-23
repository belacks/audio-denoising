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


class GaussianSmearing(nn.Module):
    def __init__(
        self,
        start: float = 0.0,
        stop: float = 1.0,
        num_gaussians: int = 6,
    ):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class DownConvBlock1d(nn.Module):
    @auto_save_hyperparams
    def __init__(self, ip_sz, op_sz, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv1d(
            ip_sz, op_sz, 
            kernel_size=kernel_size, stride=stride, padding=padding)
    def forward(self, x):
        return self.conv(x).relu()

class UpConvBlock1d(nn.Module):
    @auto_save_hyperparams
    def __init__(self, ip_sz, op_sz, kernel_size, stride, padding, last=False):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            ip_sz, op_sz, 
            kernel_size=kernel_size, padding=padding, stride=stride)
        self.last=last
    def forward(self, x, s, output_size=None):
        #x: 
        x = self.conv(x, output_size=output_size)
        if not self.last:
            ret = torch.cat((x.relu(), s), -2)
            return ret
        ret = x
        return ret

class DownBlocks(nn.Module):
    @auto_save_hyperparams
    def __init__(self, 
                 in_size, hidden_sizes, output_size,
                 kernel_sizes, 
                 strides,
                 paddings,
                 num_gaussians=6,
                 return_samples=True,
                ):
        super().__init__()
        assert in_size==1 or not return_samples
        sizes = [in_size]+list(hidden_sizes)
        #for i in range(len(sizes)-1):
        #    sizes[i]=sizes[i]
        self.downs = nn.ModuleList()
        self.len = len(sizes)-1
        self.return_samples=return_samples
        self.num_gaussians=num_gaussians
        self.gs = GaussianSmearing(num_gaussians=num_gaussians)
        for i in range(self.len):
            down=DownConvBlock1d(
                sizes[i]+num_gaussians, 
                sizes[i+1],
                kernel_sizes[i],
                strides[i],
                paddings[i],
            )
            self.downs.append(down)
    def forward(self, x):
        #old_x_shape = x.shape
        batch_size = x.size(0)
        if x.dim()==2:
            x=x.unsqueeze(2).permute(0,2,1)
        elif x.dim()!=3:
            raise Exception(f"unknown!! {x.shape}")
        # batch_size, 1, num_bins
        if self.return_samples:
            res = [x]
            for down in self.downs:
                num_bins = res[-1].size(-1)
                smear = self.gs(
                    torch.linspace(0,1, num_bins).to(self.gs.offset.device)
                ).unsqueeze(1)
                smear=smear.broadcast_to(num_bins, batch_size, -1).permute(1, 2, 0)
                informed_x = torch.cat((res[-1], smear), -2)
                res.append(down(informed_x))
        else:
            
            res = x
            for down in self.downs:
                num_bins = res.size(-1)
                smear = self.gs(
                    torch.linspace(0,1, num_bins).to(self.gs.offset.device)
                ).unsqueeze(1)
                smear=smear.broadcast_to(num_bins, batch_size, -1).permute(1, 2, 0)
                informed_x = torch.cat((res, smear), -2)
                res=(down(informed_x))
                
        return res
            
class UpBlocks(nn.Module):
    @auto_save_hyperparams
    def __init__(self, 
                 in_size, hidden_sizes, output_size,
                 kernel_sizes, 
                 strides,
                 paddings,
                 num_gaussians=6):
        super().__init__()
        assert in_size==1
        sizes = [output_size, *hidden_sizes]
        self.ups = nn.ModuleList()
        self.len = len(sizes)-1
        for i in range(self.len):#0...L-1
            up=UpConvBlock1d(
                sizes[::-1][i]+num_gaussians if i==0 else sizes[::-1][i]*2+num_gaussians, 
                sizes[::-1][i+1],
                kernel_sizes[::-1][i],
                strides[::-1][i],
                paddings[::-1][i],
            )
            self.ups.append(up)
        self.ups[-1].last=True
        self.num_gaussians=num_gaussians
        self.gs = GaussianSmearing(num_gaussians=num_gaussians)
    def forward(self, res):
        h = res[-1]
        # h: batch_size, hidden_dim, num_compressed_bins
        for i in range(self.len):
            s = res[self.len-1-i]
            output_size=(h.shape[0], h.shape[1], s.shape[-1])
            batch_size = h.shape[0]
            num_bins = h.size(-1)
            smear = self.gs(
                torch.linspace(0,1, num_bins).to(self.gs.offset.device)
            ).unsqueeze(1)
            smear=smear.broadcast_to(num_bins, batch_size, -1).permute(1, 2, 0)
            #smear: batch_size, num_gaussians, num_bins
            informed_h = torch.cat((h, smear), -2)
            h = self.ups[i](informed_h, s, output_size=output_size)
        return h

    
class GRUUNetCell(nn.Module):
    @auto_save_hyperparams
    def __init__(self, 
                 in_size, hidden_sizes, 
                 kernel_sizes, 
                 strides,
                 paddings,
                 num_gaussians=6):
        super().__init__()
        hs2=list(hidden_sizes)
        hs2[-1]=3*hs2[-1]
        self.input_gate = DownBlocks(in_size, hs2, None,
                 kernel_sizes, 
                 strides,
                 paddings,
                 num_gaussians=num_gaussians)
        self.reset_gate = DownBlocks(hidden_sizes[-1], [hs2[-1]], None,
                 [3], 
                 [1],
                 [1],
                 num_gaussians=num_gaussians, return_samples=False)
        self.output_gate = UpBlocks(in_size, hidden_sizes, 1,
                 kernel_sizes, 
                 strides,
                 paddings,
                 num_gaussians=num_gaussians)
    def forward(self, x, hx):
        # x: batch_size, input_size
        # hx: batch_size, hidden_dim, num_compressed_bins
        gate_x = self.input_gate(x)
        gate_h = self.reset_gate(hx)
        
        i_r, i_i, i_n = gate_x[-1].chunk(3,1)
        h_r, h_i, h_n = gate_h.chunk(3,1)
        inputgate = (i_i + h_i).sigmoid()
        resetgate = (i_r + h_r).sigmoid()
        newgate = (i_n + (resetgate * h_n)).tanh()

        hi = newgate + inputgate * (hx - newgate)
        out = self.output_gate(gate_x[:-1]+[hi])
        out = out.squeeze(-2)

        return out, hi
    
class GRUUNet(nn.Module):
    @auto_save_hyperparams
    def __init__(self, 
                 num_compressed_bins,
                 in_size, hidden_sizes, 
                 kernel_sizes, 
                 strides,
                 paddings,
                 num_gaussians=6,
                ):
        super().__init__()
        assert in_size==1
        self.latent_size=hidden_sizes[-1]
        self.num_compressed_bins=num_compressed_bins
        self.cell = GRUUNetCell(in_size, hidden_sizes, 
                 kernel_sizes, 
                 strides,
                 paddings,
                 num_gaussians=num_gaussians)
        
    def _gruunet(self, x, hx):
        bs, seq_len, input_size = x.size()

        outputs = []

        for x_t in x.unbind(1):
            #x_t: batch_size, input_size
            out, hx = self.cell(
                x_t,
                hx
            )
            outputs.append(out)

        outputs = torch.stack(outputs, dim=1)

        return outputs, hx

    def forward(self, input, hx=None):
        two_dimmed=input.dim() == 2
        if two_dimmed:
            input = input.unsqueeze(0)
        if hx is None:
            hx = torch.zeros(
                input.size(0),
                self.latent_size,
                self.num_compressed_bins,
                dtype=input.dtype,
                device=input.device,
            )
        
        a,b= self._gruunet(input, hx)
        if two_dimmed:
            a=a.squeeze(0)
        return a,b

