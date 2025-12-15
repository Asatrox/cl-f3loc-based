# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

from torch import Tensor
from torch import nn
import torch.nn.functional as F
import torch

try:
    from flash_attn.flash_attn_interface import flash_attn_func
    HAS_FLASH_ATTN = True
except Exception:
    HAS_FLASH_ATTN = False

XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
        use_flash_attn: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn
        self.use_flash_attn = use_flash_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None) -> Tensor:
        B, N, C = x.shape

        # [B, N, 3 * num_heads * head_dim] -> [B, 3, H, N, D]
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(0, 2, 3, 1, 4)
        )  # [B, 3, H, N, D]
        q, k, v = qkv.unbind(1)  # 各自 [B, H, N, D]

        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        # ---- 这里开始分三种情况：flash-attn / pytorch SDPA / 手写 ----
        if (
                self.use_flash_attn
                and HAS_FLASH_ATTN
                and x.is_cuda
                and x.dtype in (torch.float16, torch.bfloat16)
        ):
            # flash_attn_func 期望 [B, N, H, D]
            q_f = q.transpose(1, 2)  # [B, N, H, D]
            k_f = k.transpose(1, 2)
            v_f = v.transpose(1, 2)

            x = flash_attn_func(
                q_f,
                k_f,
                v_f,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=False,
            )  # [B, N, H, D]

            x = x.transpose(1, 2)  # [B, H, N, D]

        elif self.fused_attn:
            # PyTorch 自带的 fused SDPA（可能已经用到内部的 flash/SDPA 实现）
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )  # [B, H, N, D]
        else:
            # 纯手写注意力
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # [B, H, N, N]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v  # [B, H, N, D]

        # [B, H, N, D] -> [B, N, C]
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
