import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections  import OrderedDict

from transformers import PreTrainedModel

from .configuration_dalle imoprt DallEConfig


class Conv2d(nn.Module):
    def __init__(self, n_in, n_out, kw, config, use_float16=True):
        super().__init__()
        
        assert n_in >= 1
        assert n_out >= 1
        assert kw >= 1 and kw % 2 == 1
        
        self.n_in = n_in
        self.n_out = n_out
        self.kw = kw
        self.config = config
        self.use_float16 = use_float16
        w = torch.empty(
            (n_out, n_in, kw, kw),
            dtype=torch.float32,
            device=config.device,
            requires_grad=config.requires_grad,
        )
        w.normal_(std=1 / math.sqrt(n_in * kw ** 2))
        
        b = torch.zeros(
            (n_out,),
            dtype=torch.float32,
            device=config.device,
            requires_grad=config.requires_grad,
        )
        
        self.w = nn.Parameter(w)
        self.b = nn.Parameter(b)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_float16 and 'cuda' in self.w.device.type:
            if x.dtype != torch.float16:
                x = x.half()
            w, b = self.w.half(), self.b.half()
        else:
            if x.dtype != torch.float32:
                x = x.float()
            w, b = self.w, self.b
        return F.conv2d(x, w, b, padding=(self.kw - 1) // 2)
    
    def extra_repr(self):
        inner_repr = f"n_in={self.n_in}, n_out={self.n_out}, kw={self.kw}, "
        inner_repr += f"use_float16={self.use_float16}, "
        inner_repr += f"device={self.config.device}, "
        inner_repr += f"requires_grad={self.config.requires_grad}"
        return inner_repr
    
        
class EncoderBlock(nn.Module):
    def __init__(self, n_in, n_out, n_layers, config):
        super().__init__()
        
        assert n_in >= 1
        assert n_out >= 1 and n_out % 4 == 0
        assert n_layers >= 1
        
        self.n_in = n_in
        self.n_out = n_out
        self.n_hid = n_out // 4
        self.post_gain = 1 / (n_layers ** 2)
        
        if self.n_in != self.n_out:
            self.id_path = Conv2d(self.n_in, self.n_out, 1, config)
        else:
            self.id_path = nn.Identity()
            
        self.res_path = nn.Sequential(OrderedDict([
            ('relu_1', nn.ReLU()),
            ('conv_1', Conv2d(self.n_in, self.n_hid, 3, config)),
            ('relu_2', nn.ReLU()),
            ('conv_2', Conv2d(self.n_hid, self.n_hid, 3, config)),
            ('relu_3', nn.ReLU()),
            ('conv_3', Conv2d(self.n_hid, self.n_hid, 3, config)),
            ('relu_4', nn.ReLU()),
            ('conv_4', Conv2d(self.n_hid, self.n_out, 1, config)),
        ]))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.id_path(x) + self.post_gain * self.res_path(x)
        
        
class DallEPreTrainedModel(PreTrainedModel):
    config_class = DallEConfig
    base_model_prefix="dalle"
    
    
class DallEEncoder(DallEPreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        blk_range = range(config.n_blk_per_group)
        n_layers = config.group_count * config.n_blk_per_group
        
        in_channels = config.input_channels
        n_hid = config.n_hid
        
        self.blocks = nn.Sequential(OrderedDict([
            ('input', Conv2d(in_channels, n_hid, 7, config)),
            ('group_1', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', 
                   EncoderBlock(n_hid, n_hid, n_layers, config))
                  for i in blk_range],
                ('pool', nn.MaxPool2d(kernel_size=2)),
            ]))),
            ('group_2', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', 
                   EncoderBlock(
                       n_hid if i == 0 else 2 * n_hid, 
                       2 * n_hid, n_layers, config))
                  for i in blk_range],
                ('pool', nn.MaxPool2d(kernel_size=2)),
            ]))),
            ('group_3', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', 
                   EncoderBlock(
                       2 * n_hid if i == 0 else 4 * n_hid, 
                       4 * n_hid, n_layers, config))
                  for i in blk_range],
                ('pool', nn.MaxPool2d(kernel_size=2)),
            ]))),
            ('group_4', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}',
                   EncoderBlock(
                       4 * n_hid if i == 0 else 8 * n_hid, 
                       8 * n_hid, n_layers, config))
                  for i in blk_range],
            ]))),
            ('output', nn.Sequential(OrderedDict([
                ('relu', nn.ReLU()),
                ('conv', Conv2d(
                    8 * n_hid, config.vocab_size, 
                    1, config, use_float16=False)),
            ]))),
        ]))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 4:
            raise ValueError(f'input shape {x.shape} is not 4d')
        if x.shape[1] != self.input_channels:
            raise ValueError(f'input has {x.shape[1]} channels but model built for {self.input_channels}')
        if x.dtype != torch.float32:
            raise ValueError('input must have dtype torch.float32')
            
        return self.blocks(x)
