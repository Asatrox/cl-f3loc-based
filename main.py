import os
import cv2
import numpy as np
import torch

from modules.mono.utils.projection import warp_image_to_ground_bev

def load_images_and_txt(image_dir, txt_path):
    # 1. 读取所有图片文件，排序
    img_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    # 2. 读取 txt 的每一行
    with open(txt_path, 'r') as f:
        txt_lines = [line.strip() for line in f.readlines()]

    # 确保数量一致
    assert len(img_files) == len(txt_lines), \
        f"图片数量({len(img_files)}) 与 txt 行数({len(txt_lines)}) 不一致!"

    img_path_list = []
    data_list = []

    # 3. 逐行加载
    for i, img_name in enumerate(img_files):
        img_path = os.path.join(image_dir, img_name)
        img_path_list.append(img_path)

        data = np.array(list(map(float, txt_lines[i].split())))
        data_list.append(data)

    return data_list, img_path_list

def plot_equal_width_depths_on_bev(
    bev_img, depths,
    x_min=-5.0, x_max=5.0,
    y_min=0.1, y_max=15.0
):
    """
    bev_img: (H, W, 3) numpy 图像，RGB 或 BGR 都行
    depths: 长度为 40 的 1D 数组/列表，每个是对应的一条射线的真实深度(米)
    """
    H_bev, W_bev = bev_img.shape[:2]
    depths = np.array(depths, dtype=np.float32)
    N = len(depths)

    # 1. 等宽度 → BEV 横向 X 坐标
    X = np.linspace(x_min, x_max, N)  # (N,)

    # 2. Y 就是深度值
    Y = depths  # (N,)

    # 3. 物理坐标 -> 像素坐标
    u = (X - x_min) / (x_max - x_min) * (W_bev - 1)
    v = (Y - y_min) / (y_max - y_min) * (H_bev - 1)

    # 4. 画到图像上
    for ui, vi in zip(u, v):
        ui = int(round(ui))
        vi = int(round(vi))
        if 0 <= ui < W_bev and 0 <= vi < H_bev:
            cv2.circle(bev_img, (ui, vi), 2, (0, 0, 255), -1)  # BGR 红点

    return bev_img

if __name__ == "__main__":
    # ------------------- 配置路径 -------------------
    image_dir = "./dataset/Spencerville/rgb"
    txt_path = "./dataset/Spencerville/depth40.txt"
    save_dir = "./res"

    os.makedirs(save_dir, exist_ok=True)

    # 读取所有 (depth, img_path)
    data_list, img_path_list = load_images_and_txt(image_dir, txt_path)

    # ------------------- 常量配置 -------------------
    fx = fy = 240.0
    H_cam = 1.5

    # 地面物理范围
    x_min, x_max = -4.0, 4.0
    z_min, z_max = 0.1, 15.0

    H_bev, W_bev = 256, 256

    # ------------------- 遍历每一对 (img, depth) -------------------
    for idx, (img_path, depth) in enumerate(zip(img_path_list, data_list)):
        print(f"[{idx+1}/{len(img_path_list)}] 处理: {img_path}")

        # 1. 读取图片并转为 torch 格式
        bgr = cv2.imread(img_path)
        if bgr is None:
            print(f"  !! 无法读取图片: {img_path}")
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(rgb).float() / 255.0       # (H,W,3)
        img = img.permute(2, 0, 1).unsqueeze(0)           # (1,3,H,W)

        H_img, W_img = rgb.shape[:2]

        cx = W_img / 2.0
        cy = H_img / 2.0

        K = torch.tensor([
            [fx, 0,   cx],
            [0,  fy,  cy],
            [0,  0,   1.0],
        ], dtype=torch.float32)

        # 这里的 R 仍然用你之前的写法（注意它不是严格意义上的旋转矩阵）
        R = torch.tensor([
            [1, 0,      0],
            [0, 0,  H_cam],
            [0, 1,      0],
        ], dtype=torch.float32)

        M = K @ R  # (3,3)

        # 2. warp 到 BEV
        bev_torch, _ = warp_image_to_ground_bev(
            img, M,
            x_min, x_max,
            z_min, z_max,
            H_bev, W_bev
        )

        # 3. 转回 numpy(BGR)，方便用 OpenCV 画+保存
        bev_np = bev_torch.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H,W,3) RGB
        bev_np = (bev_np * 255.0).clip(0, 255).astype(np.uint8)
        bev_bgr = cv2.cvtColor(bev_np, cv2.COLOR_RGB2BGR)

        # 4. 把 40 条深度数据按等宽度画到 BEV 上
        bev_bgr = plot_equal_width_depths_on_bev(
            bev_bgr,
            depth,
            x_min=x_min, x_max=x_max,
            y_min=z_min, y_max=z_max
        )

        # 5. 保存到 ./res 下，文件名沿用原图名
        base_name = os.path.basename(img_path)
        save_path = os.path.join(save_dir, base_name)
        cv2.imwrite(save_path, bev_bgr)

    print("全部处理完成，结果已保存到 ./res 文件夹。") 


