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


class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = 0.0,
        stop: float = 1.0,
        num_gaussians: int = 24,
    ):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = 0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(-self.coeff * torch.pow(dist, 2))

class GRUBase(nn.RNNBase):
    """A Base module for GRU. Inheriting from GRUBase enables compatibility with torch.compile."""

    def __init__(self, *args, **kwargs):
        return super().__init__("GRU", *args, **kwargs)


for attr in nn.GRU.__dict__:
    if attr != "__init__":
        setattr(GRUBase, attr, getattr(nn.GRU, attr))
        
class GRU(GRUBase):
    @auto_save_hyperparams
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        device=None,
        dtype=None,
    ) -> None:

        if bidirectional:
            raise NotImplementedError(
                "Bidirectional LSTMs are not supported yet in this implementation."
            )

        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=False,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _gru_cell(x, hx, weight_ih, bias_ih, weight_hh, bias_hh):
        # Expected tensor shapes:
        # x:         (batch_size, input_size)
        # hx:        (batch_size, hidden_size)
        # weight_ih: (3 * hidden_size, input_size)  -> Projects input to three gates (reset, update, new candidate)
        # bias_ih:   (3 * hidden_size)
        # weight_hh: (3 * hidden_size, hidden_size) -> Projects hidden state to three gates
        # bias_hh:   (3 * hidden_size)
    
        # Ensure x is 2D [batch_size, input_size]
        x = x.view(-1, x.size(1))
    
        # Linear transformations for input and hidden state
        # Resulting shapes: (batch_size, 3 * hidden_size)
        gate_x = F.linear(x, weight_ih, bias_ih)
        gate_h = F.linear(hx, weight_hh, bias_hh)
    
        # Split the linear outputs into three chunks along the feature dimension.
        # Each chunk corresponds to one gate and has shape: (batch_size, hidden_size)
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
    
        # Compute the gates using element-wise operations:
        # resetgate: (batch_size, hidden_size)
        resetgate = (i_r + h_r).sigmoid()
        # inputgate (often referred as update gate): (batch_size, hidden_size)
        inputgate = (i_i + h_i).sigmoid()
        # newgate: (batch_size, hidden_size)
        newgate = (i_n + (resetgate * h_n)).tanh()
    
        # Compute the new hidden state:
        # hy: (batch_size, hidden_size)
        hy = newgate + inputgate * (hx - newgate)
    
        return hy


    def _gru(self, x, hx):

        if not self.batch_first:
            x = x.permute(
                1, 0, 2
            )  # Change (seq_len, batch, features) to (batch, seq_len, features)

        bs, seq_len, input_size = x.size()
        h_t = list(hx.unbind(0))

        weight_ih = []
        weight_hh = []
        bias_ih = []
        bias_hh = []
        for layer in range(self.num_layers):

            # Retrieve weights
            weights = self._all_weights[layer]
            weight_ih.append(getattr(self, weights[0]))
            weight_hh.append(getattr(self, weights[1]))
            if self.bias:
                bias_ih.append(getattr(self, weights[2]))
                bias_hh.append(getattr(self, weights[3]))
            else:
                bias_ih.append(None)
                bias_hh.append(None)

        outputs = []

        for x_t in x.unbind(1):
            for layer in range(self.num_layers):
                h_t[layer] = self._gru_cell(
                    x_t,
                    h_t[layer],
                    weight_ih[layer],
                    bias_ih[layer],
                    weight_hh[layer],
                    bias_hh[layer],
                )

                # Apply dropout if in training mode and not the last layer
                if layer < self.num_layers - 1 and self.dropout:
                    x_t = F.dropout(h_t[layer], p=self.dropout, training=self.training)
                else:
                    x_t = h_t[layer]

            outputs.append(x_t)

        outputs = torch.stack(outputs, dim=1)
        if not self.batch_first:
            outputs = outputs.permute(
                1, 0, 2
            )  # Change back (batch, seq_len, features) to (seq_len, batch, features)

        return outputs, torch.stack(h_t, 0)

    def forward(self, input, hx=None):  # noqa: F811
        if input.dim() != 3:
            raise ValueError(
                f"GRU: Expected input to be 3D, got {input.dim()}D instead"
            )
        if hx is not None and hx.dim() != 3:
            raise RuntimeError(
                f"For batched 3-D input, hx should also be 3-D but got {hx.dim()}-D tensor"
            )
        max_batch_size = input.size(0) if self.batch_first else input.size(1)
        if hx is None:
            hx = torch.zeros(
                self.num_layers,
                max_batch_size,
                self.hidden_size,
                dtype=input.dtype,
                device=input.device,
            )

        self.check_forward_args(input, hx, batch_sizes=None)
        result = self._gru(input, hx)

        output = result[0]
        hidden = result[1]

        return output, hidden