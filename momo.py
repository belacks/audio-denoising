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
        self.conv = nn.Conv1d(ip_sz, op_sz, kernel_size=kernel_size, stride=stride, padding=padding)
    def forward(self, x):
        #print("    DownConv:Conv1d",x.shape)
        return self.conv(x).relu()

class UpConvBlock1d(nn.Module):
    @auto_save_hyperparams
    def __init__(self, ip_sz, op_sz, kernel_size, stride, padding, last=False):
        super().__init__()
        self.conv = nn.ConvTranspose1d(ip_sz, op_sz, kernel_size=kernel_size, padding=padding, stride=stride)
        self.last=last
    def forward(self, x, s, output_size=None):
        #print("    UpConvBlock1d:ConvTranspose1d",x.shape, output_size)
        x = self.conv(x, output_size=output_size)
        if not self.last:
            #print("    UpConvBlock1d:cat",x.shape, s.shape)
            return torch.cat((x.relu(), s), -2)
        return x

class UNet(nn.Module):
    @auto_save_hyperparams
    def __init__(self, 
                 in_size, hidden_sizes, output_size,
                 kernel_sizes, 
                 strides,
                 paddings,
                 num_gaussians=6):
        super().__init__()
        assert in_size==1
        sizes = [in_size+num_gaussians, *hidden_sizes]
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.len = len(sizes)-1
        for i in range(self.len):#0...L-1
            
            down=DownConvBlock1d(
                sizes[i], 
                sizes[i+1],
                kernel_sizes[i],
                strides[i],
                paddings[i]
            )
            
            up=UpConvBlock1d(
                sizes[::-1][i] if i==0 else 2*sizes[::-1][i], 
                sizes[::-1][i+1],
                kernel_sizes[::-1][i],
                strides[::-1][i],
                paddings[::-1][i]
            )
                
            self.downs.append(down)
            self.ups.append(up)
        self.ups[-1]= UpConvBlock1d(
            2*sizes[::-1][i], 
            output_size,
            kernel_sizes[::-1][i],
            strides[::-1][i],
            paddings[::-1][i]
        ) 
        self.ups[-1].last=True
        self.ups[-1].last=True
        self.gs = GaussianSmearing(num_gaussians=num_gaussians)
        
    def forward(self, x):
        # input: N, C
        # outputs: N, O, C
        old_x_shape = x.shape
        batch_size = x.size(0)
        num_bins = x.size(1)
        assert (x.shape)==(batch_size, num_bins)
        x=x.unsqueeze(2).permute(1,2,0) # num_bins, 1, batch_size
        
        smear = self.gs(
            torch.linspace(0,1, num_bins).to(self.gs.offset.device)
        ).unsqueeze(1)
        smear=smear.broadcast_to(num_bins, batch_size, -1).permute(0, 2, 1)
        #print("  UNet:cat", x.shape, smear.shape)
        informed_x = torch.cat((x, smear), -2).permute(2,1,0) # batch_size, 1+G, num_bins
        res = [informed_x]
        for down in self.downs:
            #print("  UNet:DownConvBlock1d", res[-1].shape)
            res.append(down(res[-1]))
        h = res[-1]
        for i in range(self.len):
            s = res[self.len-1-i]
            output_size=(h.shape[0], h.shape[1], s.shape[-1])
            #print("  UNet:UpConvBlock1d", h.shape, s.shape, output_size)
            h = self.ups[i](h, s, output_size=output_size)
        return h
        
    
class MOMOCell(nn.Module):
    @auto_save_hyperparams
    def __init__(self, 
                 in_size, hidden_sizes, 
                 kernel_sizes, 
                 strides,
                 paddings,
                 num_gaussians=6):
        super().__init__()
        self.input_gate = UNet(in_size, hidden_sizes, 3,
                 kernel_sizes, 
                 strides,
                 paddings,
                 num_gaussians=num_gaussians)
        self.reset_gate = UNet(in_size, hidden_sizes, 3,
                 kernel_sizes, 
                 strides,
                 paddings,
                 num_gaussians=num_gaussians)
        #can we combine these gates into 1 as in M-GRU?
    def forward(self, x, hx):
        #batch_size, num_bins
        #print("MOMOCell:input_gate",x.shape)
        gate_x = self.input_gate(x)
        #print("MOMOCell:reset_gate",hx.shape)
        gate_h = self.reset_gate(hx)

        i_r, i_i, i_n = gate_x.unbind(1)
        h_r, h_i, h_n = gate_h.unbind(-2)#1 and -2 should be the same?
        
        inputgate = (i_i + h_i).sigmoid()#update gate
        resetgate = (i_r + h_r).sigmoid()
        newgate = (i_n + (resetgate * h_n)).tanh()

        hy = newgate + inputgate * (hx - newgate)

        return hy
    
class MOMO(nn.Module):
    @auto_save_hyperparams
    def __init__(self, 
                 num_bins,
                 in_size, hidden_sizes, 
                 kernel_sizes, 
                 strides,
                 paddings,
                 num_gaussians=6):
        super().__init__()
        assert in_size==1
        self.num_bins=num_bins
        self.cell = MOMOCell(in_size, hidden_sizes, 
                 kernel_sizes, 
                 strides,
                 paddings,
                 num_gaussians=num_gaussians)
        self.output_gate = UNet(in_size, hidden_sizes, in_size,
                 kernel_sizes, 
                 strides,
                 paddings,
                 num_gaussians=num_gaussians)
        
    def _momo(self, x, hx):
        bs, seq_len, input_size = x.size()

        outputs = []

        for x_t in x.unbind(1):
            #print("MOMO:cell",x_t.shape,hx.shape)
            hx = self.cell(
                x_t,
                hx
            )
            outputs.append(self.output_gate(hx).squeeze(-2))

        outputs = torch.stack(outputs, dim=1)

        return outputs, hx

    def forward(self, input, hx=None):
        two_dimmed=input.dim() == 2
        if two_dimmed:
            input = input.unsqueeze(0)
        #input: batch, seq_len, features
        #if input.dim() != 2:
        #    raise ValueError(
        #        f"MOMO: Expected input to be 2D, got {input.dim()}D instead"
        #    )
        #if hx is not None and hx.dim() != 3:
        #    raise RuntimeError(
        #        f"For batched 3-D input, hx should also be 3-D but got {hx.dim()}-D tensor"
        #    )
        if hx is None:
            hx = torch.zeros(
                input.size(0),
                self.num_bins,
                dtype=input.dtype,
                device=input.device,
            )
        
        a,b= self._momo(input, hx)
        if two_dimmed:
            a=a.squeeze(0)
        return a,b

