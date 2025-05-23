#momo3: added first-order delta
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
        #print("      DownConv:Conv1d",x.shape)
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
        #print("      UpConvBlock1d:ConvTranspose1d",x.shape, output_size)
        x = self.conv(x, output_size=output_size)
        if not self.last:
            #print("      UpConvBlock1d:cat",x.shape, s.shape)
            ret = torch.cat((x.relu(), s), -2)
            #print("      UpConvBlock1d:ret:",ret.shape)
            return ret
        ret = x
        #print("      UpConvBlock1d:ret:",ret.shape)
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
        sizes = [in_size+num_gaussians, *hidden_sizes]
        self.downs = nn.ModuleList()
        self.len = len(sizes)-1
        self.return_samples=return_samples
        for i in range(self.len):#0...L-1
            down=DownConvBlock1d(
                sizes[i], 
                sizes[i+1],
                kernel_sizes[i],
                strides[i],
                paddings[i]
            )
            self.downs.append(down)
        self.gs = GaussianSmearing(num_gaussians=num_gaussians)
    def forward(self, x):
        batch_size = x.size(0)
        num_bins = x.size(-1)
        if x.dim()==2:
            #assert (x.shape)==(batch_size, num_bins)
            x=x.unsqueeze(2).permute(1,2,0) # num_bins, 1, batch_size
        elif x.dim()==3:
            #x: [batch_size, hidden_size, num_bins]
            x=x.permute(2,1,0)
            #x: [num_bins, hidden_size, batch_size]
        else:
            raise Exception(f"unknown!! {x.shape}")
        smear = self.gs(
            torch.linspace(0,1, num_bins).to(self.gs.offset.device)
        ).unsqueeze(1)
        smear=smear.broadcast_to(num_bins, batch_size, -1).permute(0, 2, 1)
        #print("    UNet:cat", x.shape, smear.shape)
        informed_x = torch.cat((x, smear), -2).permute(2,1,0) # batch_size, 1+G, num_bins
        if self.return_samples:
            res = [informed_x]
            for down in self.downs:
                #print("    UNet:DownConvBlock1d", res[-1].shape)
                res.append(down(res[-1]))
        else:
            
            res = informed_x
            for down in self.downs:
                #print("    UNet:DownConvBlock1d", res[-1].shape)
                res=(down(res))
                
        return res
            
class UpBlocks(nn.Module):
    @auto_save_hyperparams
    def __init__(self, 
                 in_size, hidden_sizes, output_size,
                 kernel_sizes, 
                 strides,
                 paddings):
        super().__init__()
        sizes = [output_size, *hidden_sizes]
        self.ups = nn.ModuleList()
        self.len = len(sizes)-1
        for i in range(self.len):#0...L-1
            up=UpConvBlock1d(
                sizes[::-1][i] if i==0 else 2*sizes[::-1][i], 
                sizes[::-1][i+1],
                kernel_sizes[::-1][i],
                strides[::-1][i],
                paddings[::-1][i]
            )
            self.ups.append(up)
        self.ups[-1].last=True
    def forward(self, res):
        h = res[-1]
        for i in range(self.len):
            s = res[self.len-1-i]
            output_size=(h.shape[0], h.shape[1], s.shape[-1])
            #print("    UNet:UpConvBlock1d", h.shape, s.shape, output_size)
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
        hs2=list(hidden_sizes)
        hs2[-1]=3*hs2[-1]
        self.input_gate = DownBlocks(in_size, hs2, None,
                 kernel_sizes, 
                 strides,
                 paddings,
                 num_gaussians=num_gaussians)
        #hidden_sizes=list(hidden_sizes)
        #hidden_sizes[-1]=3*hidden_sizes[-1]
        self.reset_gate = DownBlocks(hidden_sizes[-1], [3*hidden_sizes[-1]], None,
                 [3], 
                 [1],
                 [1],
                 num_gaussians=num_gaussians, return_samples=False)
        self.output_gate = UpBlocks(in_size, hidden_sizes, 1,
                 kernel_sizes, 
                 strides,
                 paddings)
        #can we combine these gates into 1 as in M-GRU?
    def forward(self, x, hx):
        #batch_size, num_bins
        #print("  MOMOCell:input_gate",x.shape)
        gate_x = self.input_gate(x)
        #print("  MOMOCell:gate_x[-1]:",gate_x[-1].shape)
        #print("  MOMOCell:reset_gate",hx.shape)
        gate_h = self.reset_gate(hx)
        #print("  MOMOCell:gate_h:",gate_h.shape)
        #N, 3*C, L
        i_r, i_i, i_n = gate_x[-1].chunk(3,1)
        h_r, h_i, h_n = gate_h.chunk(3,1)
        #print("  MOMOCell:i_i", i_i.shape)
        #print("  MOMOCell:h_i", h_i.shape)
        inputgate = (i_i + h_i).sigmoid()#update gate
        resetgate = (i_r + h_r).sigmoid()
        newgate = (i_n + (resetgate * h_n)).tanh()
        #print("  MOMOCell:hx:",hx.shape)
        #print("  MOMOCell:newgate:",newgate.shape)
        #print("  MOMOCell:inputgate:",inputgate.shape)

        hi = newgate + inputgate * (hx - newgate)
        #print("  MOMOCell:hi:",hi.shape)
        out = self.output_gate(gate_x[:-1]+[hi])
        out = out.squeeze(-2)
        #print("  MOMOCell:out:",out.shape)

        return out, hi
    
class MOMO3(nn.Module):
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
        self.latent_size=hidden_sizes[-1]
        self.num_compressed_bins=num_compressed_bins
        self.cell = MOMOCell(in_size+1, hidden_sizes, 
                 kernel_sizes, 
                 strides,
                 paddings,
                 num_gaussians=num_gaussians)
        #self.output_gate = UNet(in_size, hidden_sizes, in_size,
        #         kernel_sizes, 
        #         strides,
        #         paddings,
        #         num_gaussians=num_gaussians)
    def _momo(self, x, hx, prev=None):
        bs, seq_len, input_size = x.size()

        outputs = []
        
        for x_t in x.unbind(1):
                
            if x_t.dim()==2:
                #assert (x.shape)==(batch_size, num_bins)
                x_t=x_t.unsqueeze(1)
            if prev is None:
                prev=x_t.detach().clone()
            #print("MOMO:cell",x_t.shape,hx.shape)
            # at first, x: [batch_size, num_bins]
            # we want: #x: [batch_size, hidden_size, num_bins]
            out, hx = self.cell(
                torch.cat([x_t,x_t-prev],-2),
                hx
            )
            prev=x_t.detach().clone()
            #print("MOMO:out:",out.shape)
            outputs.append(out)

        outputs = torch.stack(outputs, dim=1)

        return outputs, hx

    def forward(self, input, hx=None, prev=None):
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
                self.latent_size,
                self.num_compressed_bins,
                dtype=input.dtype,
                device=input.device,
            )
        
        a,b= self._momo(input, hx, prev=prev)
        if two_dimmed:
            #print("a is two dimmed")
            a=a.squeeze(0)
        return a,b

