"""
vggt_obs.py
-------------------------------
包含VGGT模型特征提取的相关代码
"""
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from modules.vggt.layers.vision_transformer import vit_large
from modules.vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from modules.vggt.layers.block import Block

# 图像增强的参数
_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

"""
输入：(B, S, H, W) 
    W 为 518, H 为392 
输出：交叉注意力层每一层的(B, S, P, C)特征集合 + start_idx  
    P 大小默认是 14 * 14，共1036
    C 是2048, 帧间注意力和全局注意力拼接
    
中间：1）先将 patch 转化为tokens
     2）添加camera tokens(S, 1, C)和register tokens(S, 4, C) (query: 参考帧 + others：其余帧，共享token)，tokens变为(S, 1036 + 4 + 1 = 1041, C)
     3）交叉注意力层
"""

class VGGTObs(nn.Module):
    """
    参考 VGGT 的 Aggregator 模块的代码（不考虑后面的预测头）
    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
    ):
        super().__init__()

        self.__build_patch_embed__(
            img_size=img_size,
            patch_size=patch_size
        )

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        """
        block_fn 结构说明：
        
        输入：dim = 1024
        输出：dim = 1024
        
        
        - norm1：Attention 前的 LayerNorm（Pre-LN）
        - attn：多头自注意力层
        - ls1：Attention 残差分支的 LayerScale
        - drop_path1：Attention 残差的 DropPath（或 Identity）

        - norm2：FFN 前的 LayerNorm
        - mlp：两层前馈网络 MLP（隐藏维度 = dim * mlp_ratio）
        - ls2：FFN 残差分支的 LayerScale
        - drop_path2：FFN 残差的 DropPath

        整体结构：
        x = x + DropPath( LayerScale( Attention( LayerNorm(x) ) ) )
        x = x + DropPath( LayerScale( MLP( LayerNorm(x) ) ) )
        """

        # 帧内自注意力模块
        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )
        # 全局自注意力模块
        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size
        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False  # hardcoded to False

    # Token embedding
    def __build_patch_embed__(
        self,
        img_size=518,
        patch_size=14,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
    ):
        """
        Use a vision transformer(default: vit_large) to build the patch embed layer.
        """
        self.patch_embed = vit_large(
            img_size=img_size,
            patch_size=patch_size,
            num_register_tokens=num_register_tokens,
            interpolate_antialias=interpolate_antialias,
            interpolate_offset=interpolate_offset,
            block_chunks=block_chunks,
            init_values=init_values,
        )

    def forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W] or [S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        B, S, C, H, W = images.shape

        if C != 3:
            raise ValueError(f"Expected 3 input channels, got {C}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        # P: patch size, C_P: Patch channels
        _, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length(1, 2, X, C) => (B, S, X, C)
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens (B * S, X1 + X2 + P, C)
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        # 交叉注意力层
        frame_idx = 0
        global_idx = 0
        output_list = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list, self.patch_start_idx

    # 帧间、全局自注意力计算模块（N， P， C）
    # 帧间的 N => B * S, P => P
    # 全局的 N => B, P => S * P
    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                # 用 gradient checkpoint 包裹一个 Transformer Block，执行 Block(tokens, pos) 前向，并节省显存
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates

def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """
    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    # others之间共享参数
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined

# 验证权重文件和模型正确性
if __name__ == "__main__":
    from utils.load_imgs import load_and_preprocess_imgs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

    path = "../../../checkpoints/vggt_obs.pth"

    model = VGGTObs()

    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)

    image_paths = ["../../../imgs/00000-0.png", "../../../imgs/00000-1.png", "../../../imgs/00000-2.png", "../../../imgs/00000-3.png"]
    images = load_and_preprocess_imgs(image_paths).to(device)
    print(images.shape)
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=dtype):
            outputs = model(images)
    print(outputs)

