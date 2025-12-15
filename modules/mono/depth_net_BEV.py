
"""
This is module predict the structural ray scan from perspective image
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as fn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import *

from modules.vggt.vggt_obs import VGGTObs
from modules.mono.utils.projection import *
from modules.network_utils import *


class depth_net_BEV(nn.Module):
    def __init__(self, d_min=0.1, d_max=15.0, d_hyp=-0.2, D=128, type="direct", f = 1036) -> None:
        super().__init__()
        """
        depth_feature extracts depth features:
        img (480, 640, 3) => (1, 40, 32)

        FC to predict depth:
        (1, 40, 32) => FC => (1, 40)

        Alternative: directly use the (1, 40, 128) as probability volume, supervise on softmax
        """
        self.d_min = d_min
        self.d_max = d_max
        self.d_hyp = d_hyp
        self.D = D
        self.H_BEV = 128
        self.W_BEV = 40
        self.f = f
        self.depth_feature = None
        self.depth_head = None
        if type == "direct":
            self.depth_feature = BEV_feature_direct(self.H_BEV, self.W_BEV)
            self.depth_head = BEV_depth_head_direct(in_channels=2048, hidden_channels=256)
        else:
            raise NotImplementedError(f"Unknown BEV feature type: {type}")
        

    def forward(self, x):
        # extract depth features
        x = self.depth_feature(x, self.f)  # (N, fW, D)
        x = self.depth_head(x)  # (N, fW)
        return x


class BEV_feature_direct(nn.Module):
    def __init__(self, H, W, K, R) -> None:
        super().__init__()
        self.H_bev = H
        self.W_bev = W
        self.H_cam = 1.5
        self.x_min, self.x_max = -5.0, 5.0   # 左右 10m
        self.y_min, self.y_max =  0.1, 15.0  # 从 0.1m 到 15m 远
        self.obs = VGGTObs()
        self.obs = self.__load_model__()
        self.scale = self.obs.patch_size
        self.K = K
        self.K[0, 3] = self.K[0, 3] / self.scale
        self.K[1, 3] = self.K[1, 3] / self.scale
        self.R = R
        self.R[0, 2] = self.R[0, 2] / self.scale
        self.R[1, 2] = self.R[1, 2] / self.scale
        self.R[2, 2] = self.R[2, 2] / self.scale
        self.M = self.K @ self.R

    def __load_model__(self):
        model = VGGTObs()
        path = "./checkpoints/vggt_obs.pth"
        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict)
        return model
    
    def __tokens_to_feature__(self, x, H, W):
        """
        x: (B, S, P, C)
        H, W: 还原 token 的空间大小 (P = H * W)

        返回:
            x_5d: (B, S, C, H, W)
            x_bev: (B, C, H, W)   # 取 S=0
        """
        B, S, P, C = x.shape

        # (B, S, P, C) → (B, S, H, W, C)
        x = x.view(B, S, H, W, C)

        # (B, S, H, W, C) → (B, S, C, H, W)
        x_5d = x.permute(0, 1, 4, 2, 3).contiguous()

        # 取 S=0 的那一帧
        x_rgb = x_5d[:, 0]   # (B, C, H, W)

        return x_rgb


    def forward(self, image):
        B, S, C, H_img, W_img = image.shape
        H_feat = H_img // self.scale
        W_feat = W_img // self.scale
        token_sets, start_idx = self.obs(image)
        tokens = token_sets[23]
        tokens = tokens[:, :, start_idx:, :]
        feat = self.__tokens_to_feature__(tokens, H_feat, W_feat)
        feat_bev, _ = warp_image_to_ground_bev(
            feat, self.M, self.H_cam,
            self.x_min, self.x_max,
            self.y_min, self.y_max,
            self.H_bev, self.W_bev)
        # ---- 可视化 BEV 特征 ----
        visualize_feat_map(feat_bev, "feat_bev.png")
        
        return feat_bev
    

class BEV_depth_head_direct(nn.Module):
    """
    输入:  BEV 特征 (B, C=2048, H=128, W=40)
    输出:  深度预测 (B, W=40)
    """
    def __init__(self,
                 in_channels=2048,
                 hidden_channels=256):
        super().__init__()

        # 1) 通道降维: 2048 -> 256
        self.channel_reduce = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=1
        )

        # 2) 只在 H 方向做几层卷积，加强竖直方向聚合
        self.vert_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels,
                      kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels,
                      kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
        )

        # 3) 在 H 维度上做全局池化: (B, hidden, H, W) -> (B, hidden, 1, W)
        self.pool_h = nn.AdaptiveAvgPool2d((1, None))

        # 4) 用 1x1 卷积预测每列一个标量: (B, hidden, 1, W) -> (B, 1, 1, W)
        self.pred = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, bev_feat):
        """
        bev_feat: (B, C, H, W)
        return:   (B, W)
        """
        x = self.channel_reduce(bev_feat)  # (B, hidden, H, W)
        x = self.vert_conv(x)              # (B, hidden, H, W)
        x = self.pool_h(x)                 # (B, hidden, 1, W)
        x = self.pred(x)                   # (B, 1, 1, W)
        x = x.squeeze(2).squeeze(1)        # -> (B, W)
        return x

import cv2
import numpy as np

def visualize_feat_map(feat_4d, save_path):
    """
    feat_4d: (B, C, H, W) 的特征
    save_path: 保存路径，.png/.jpg 都可以
    """
    # 只看 batch 里第 0 个
    feat = feat_4d[0]        # (C, H, W)
    feat = feat.detach().cpu()

    # 用 L2 norm 聚合通道: (C,H,W) -> (H,W)
    feat_norm = torch.norm(feat, p=2, dim=0)  # (H, W)

    # 归一化到 [0, 255]
    # mask = feat_norm > 0               # 去除 0
    # min_nonzero = feat_norm[mask].min()
    # feat_norm -= min_nonzero
    if feat_norm.max() > 0:
        feat_norm /= feat_norm.max()
    feat_img = (feat_norm.numpy() * 255.0).astype(np.uint8)  # (H, W)

    # 伪彩色，方便看
    feat_color = cv2.applyColorMap(feat_img, cv2.COLORMAP_JET)

    cv2.imwrite(save_path, feat_color)
