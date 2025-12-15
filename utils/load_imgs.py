"""
load_imgs.py
-------------------------------
1)导入图片，转化为tensor
2)调整图片大小为518 × X
3)返回 (N, 3, H, W)

注意：
图像标准化在VGGTObs模型中进行了处理
调整图片大小分成两种模式"pad"和"crop"
如果导入的多张图片尺寸不一致，选择最大尺寸，不足的进行常数 1 填充。
1)pad:
将图片等比例缩放，使得最长边为 518 。如果短边小于 518 ，用 1 进行填充。
最终得到 518 × 518 的图像

2)crop
将图像等比例缩放，使得宽为 518 。如果高大于 518， 进行中心裁剪。
"""
import torch
from PIL import Image
from torchvision import transforms as TF


def load_and_preprocess_imgs(image_path_list, mode="crop"):
    """
        This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

        Args:
            image_path_list (list): List of paths to image files
            mode (str, optional): Preprocessing mode, either "crop" or "pad".
                                 - "crop" (default): Sets width to 518px and center crops height if needed.
                                 - "pad": Preserves all pixels by making the largest dimension 518px
                                   and padding the smaller dimension to reach a square shape.

        Returns:
            torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

        Raises:
            ValueError: If the input list is empty or if mode is invalid

        Notes:
            - Images with different dimensions will be padded with white (value=1.0)
            - A warning is printed when images have different shapes
            - When mode="crop": The function ensures width=518px while maintaining aspect ratio
              and height is center-cropped if larger than 518px
            - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
              and the smaller dimension is padded to reach a square shape (518x518)
            - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
        """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("Images list must not be empty")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    for image_path in image_path_list:
        img = Image.open(image_path)

        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)

        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                # 保证是14的倍数
                new_height = round(height * (new_width / width) / 14) * 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14
        else:  # mode == "crop"
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y: start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images

# 测试不同图片导入方式
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import torch

    img_paths = ["../../imgs/00000-0.png", "../../imgs/00000-1.png", "../../imgs/00000-2.png", "../../imgs/00000-3.png"]

    imgs = load_and_preprocess_imgs(img_paths)  # shape: (C,H,W)

    # 如果是 torch.Tensor，需要转 numpy
    img_np = imgs.detach().cpu().numpy()

    # 把 (N,C,H,W) → (N,H,W,C)
    img_np = img_np.transpose(0, 2, 3, 1)
    print(img_np.shape)

    # 如果你的图是 0~1 之间的浮点数，这样显示就没问题
    plt.imshow(img_np[0])
    plt.axis('off')
    plt.show()




