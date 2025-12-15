import cv2
import torch
import kornia
import numpy as np

def warp_image_to_ground_bev(
    feat,       # 现在你是特征图，不是原始 RGB
    M,
    x_min, x_max,
    y_min, y_max,
    H_bev, W_bev
):
    device = feat.device
    B = feat.shape[0]

    M_m2p = torch.tensor([
        [(x_max - x_min) / (W_bev - 1), 0.0, x_min],
        [0.0, (y_max - y_min) / (H_bev - 1), y_min],
        [0.0, 0.0, 1.0],
    ], dtype=feat.dtype, device=device)  # (3,3)

    M = M @ M_m2p
    M = torch.inverse(M)
    M = M.expand(1, -1, -1)


    # 4. warp 特征
    bev_feat = kornia.geometry.transform.warp_perspective(
        feat,
        M,
        dsize=(H_bev, W_bev),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True,
    )

    return bev_feat, M



# 测试代码
if __name__ == "__main__":
    img_path = "imgs/00000-1.png"
    bgr = cv2.imread(img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    img = torch.from_numpy(rgb).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

    H_img, W_img = rgb.shape[:2]

    fx = fy = 800.0
    cx = W_img / 2
    cy = H_img / 2
    K = torch.tensor([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=torch.float32)

    # 相机高度（假设摄像头距离地面 1.5m）
    H_cam = 1.5

    # 地面上想截取的区域（单位:米）
    x_min, x_max = -5.0, 5.0   # 左右 10m
    z_min, z_max =  0.1, 15.0  # 从 1m 到 20m 远

    R = torch.tensor([[1, 0, 0],
                      [0, 0, H_cam],
                      [0, 1, 0]], dtype=torch.float32)
    
    M = K @ R

    H_bev, W_bev = 256, 256

    bev, _ = warp_image_to_ground_bev(
        img, M, 
        x_min, x_max,
        z_min, z_max,
        H_bev, W_bev
    )

    bev_np = bev.squeeze(0).permute(1, 2, 0).numpy()
    bev_show = (bev_np * 255).clip(0, 255).astype(np.uint8)
    bev_show = cv2.cvtColor(bev_show, cv2.COLOR_RGB2BGR)

    cv2.imshow("BEV", bev_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



