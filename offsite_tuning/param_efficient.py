import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.pytorch_utils import Conv1D


class LoRALayer():
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class LoRALinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (self.lora_dropout(x) @ self.lora_A.T @
                           self.lora_B.T) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, r: int, lora_alpha: int, lora_dropout: float, merge_weights: bool):
        # Create a LoRALinear layer from a linear layer
        lora_linear = cls(linear.in_features,
                          linear.out_features,
                          r=r,
                          lora_alpha=lora_alpha,
                          lora_dropout=lora_dropout,
                          merge_weights=merge_weights
                          )
        # Copy the weights
        lora_linear.weight.data = linear.weight.data
        if hasattr(linear, 'bias') and linear.bias is not None:
            lora_linear.bias.data = linear.bias.data
        return lora_linear

    @classmethod
    def from_conv1d(cls, conv1d: Conv1D, r: int, lora_alpha: int, lora_dropout: float, merge_weights: bool):
        """
        1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

        Basically works like a linear layer but the weights are transposed.
        """
        # Create a LoRALinear layer from a conv1d layer
        lora_linear = cls(conv1d.weight.size(0),
                          conv1d.weight.size(1),
                          r=r,
                          lora_alpha=lora_alpha,
                          lora_dropout=lora_dropout,
                          merge_weights=merge_weights
                          )
        # Copy the weights
        lora_linear.weight.data = conv1d.weight.data.T
        if hasattr(conv1d, 'bias') and conv1d.bias is not None:
            lora_linear.bias.data = conv1d.bias.data
        return lora_linear


class Adapter(nn.Module):
    def __init__(self, embed_dim: int, adapter_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, adapter_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(adapter_size, embed_dim)
        self.act_fn = nn.ReLU()

    def forward(self, x: torch.Tensor):
        residual = x
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = x + residual
        return x


def add_adapter(layer, embed_dim: int, adapter_size: int, dropout: float):
    # add an adapter module after the forward pass of a linear layer
    # register a forward hook to the layer
    def forward_hook(module, input, output):
        return module.adapter(output)
    layer.adapter = Adapter(embed_dim, adapter_size, dropout)
    layer.register_forward_hook(forward_hook)


def use_adapter(layers: nn.ModuleList, adapter_size: int, dropout: float = 0.1):
    if isinstance(layers[0], OPTDecoderLayer):
        for layer in layers:
            add_adapter(layer.self_attn.out_proj,
                        layer.embed_dim, adapter_size, dropout)
            add_adapter(layer.fc2, layer.embed_dim, adapter_size, dropout)
    elif isinstance(layers[0], GPT2Block):
        for layer in layers:
            add_adapter(layer.attn.c_proj, layer.attn.embed_dim,
                        adapter_size, dropout)
            add_adapter(layer.mlp.c_proj, layer.attn.embed_dim,
                        adapter_size, dropout)
    else:
        raise NotImplementedError

    # freeze all parameters except the adapter modules
    for param in layers.parameters():
        param.requires_grad = False
    for name, param in layers.named_parameters():
        if 'adapter' in name:
            param.requires_grad = True


def use_lora(layers: nn.ModuleList, r: int, lora_alpha: int, lora_dropout: float = 0.1, merge_weights: bool = False):
    # Replace all linear layers with LoRALinear layers
    if isinstance(layers[0], OPTDecoderLayer):
        for layer in layers:
            layer.self_attn.q_proj = LoRALinear.from_linear(
                layer.self_attn.q_proj, r, lora_alpha, lora_dropout, merge_weights)
            layer.self_attn.k_proj = LoRALinear.from_linear(
                layer.self_attn.k_proj, r, lora_alpha, lora_dropout, merge_weights)
            layer.self_attn.v_proj = LoRALinear.from_linear(
                layer.self_attn.v_proj, r, lora_alpha, lora_dropout, merge_weights)
            layer.self_attn.out_proj = LoRALinear.from_linear(
                layer.self_attn.out_proj, r, lora_alpha, lora_dropout, merge_weights)
            layer.fc1 = LoRALinear.from_linear(
                layer.fc1, r, lora_alpha, lora_dropout, merge_weights)
            layer.fc2 = LoRALinear.from_linear(
                layer.fc2, r, lora_alpha, lora_dropout, merge_weights)
    elif isinstance(layers[0], GPT2Block):
        for layer in layers:
            layer.attn.c_attn = LoRALinear.from_conv1d(
                layer.attn.c_attn, r, lora_alpha, lora_dropout, merge_weights)
            layer.attn.c_proj = LoRALinear.from_conv1d(
                layer.attn.c_proj, r, lora_alpha, lora_dropout, merge_weights)
            layer.mlp.c_fc = LoRALinear.from_conv1d(
                layer.mlp.c_fc, r, lora_alpha, lora_dropout, merge_weights)
            layer.mlp.c_proj = LoRALinear.from_conv1d(
                layer.mlp.c_proj, r, lora_alpha, lora_dropout, merge_weights)
    else:
        raise NotImplementedError

    # freeze all parameters except the LoRALinear layers
    for param in layers.parameters():
        param.requires_grad = False
    for name, param in layers.named_parameters():
        if 'lora' in name:
            param.requires_grad = True


def use_bitfit(model: nn.Module):
    # freeze all parameters except the bias terms
    # train the bias terms only
    for param in model.parameters():
        param.requires_grad = False
    for param in model.parameters():
        if param.dim() == 1:
            param.requires_grad = True
