import os
import sys
import argparse
import cv2
import random
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from skimage.measure import find_contours
import skimage.io
from transformers import ViTImageProcessor, ViTModel

# --------------------------------------------------------
# 1. 辅助可视化函数 (保持原样或微调)
# --------------------------------------------------------

def combine_images(source_folder, output_filename="combined_result.png"):
    # 准备文件列表和对应的标签
    image_list = []
    
    # 添加原始图片
    image_list.append(("img.png", "Original Image"))
    
    # ViT-Base 有 12 个头
    for i in range(12):
        filename = f"attn-head{i}.png"
        label = f"Head {i}"
        image_list.append((filename, label))
    
    total_images = len(image_list)

    # 创建画布
    fig, axes = plt.subplots(1, total_images, figsize=(24, 3))
    
    if total_images == 1:
        axes = [axes]

    for idx, (filename, label) in enumerate(image_list):
        path = os.path.join(source_folder, filename)
        ax = axes[idx]
        
        if os.path.exists(path):
            try:
                img = Image.open(path)
                ax.imshow(img)
            except Exception as e:
                print(f"无法加载图片 {filename}: {e}")
                ax.text(0.5, 0.5, "Error", ha='center', va='center')
        else:
            # print(f"警告: 文件不存在 -> {path}")
            ax.text(0.5, 0.5, "Missing", ha='center', va='center')
        
        ax.set_xlabel(label, fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"合并图片已保存为: {output_filename}")
    plt.close()

def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image

def random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()
    N = 1
    mask = mask[None, :, :]
    colors = random_colors(N)
    height, width = image.shape[:2]
    ax.set_ylim(height, 0)
    ax.set_xlim(0, width)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur: _mask = cv2.blur(_mask,(10,10))
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    # print(f"{fname} saved.")
    plt.close(fig)

# --------------------------------------------------------
# 2. 主程序
# --------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize ViT Self-Attention maps')
    parser.add_argument("--image_path", required=True, type=str, help="Path of the image to load.")
    parser.add_argument('--output_dir', default='vit_vis_output', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=0.6, help="Percentage of mass to keep.")
    # ViT Base 默认为 layer 0-11，你可以选择前几层，例如 0, 1, 2
    parser.add_argument('--layer', default=0, type=int, help='The layer index to visualize (0-11 for ViT-Base).')
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 1. 加载 Hugging Face ViT 模型
    # 使用 ViTModel 即可，因为它直接返回 hidden states 和 attentions，不需要分类头
    model_name = "google/vit-base-patch16-224"
    print(f"Loading model: {model_name}...")
    
    processor = ViTImageProcessor.from_pretrained(model_name)
    # 关键：output_attentions=True 让模型返回所有层的注意力图
    model = ViTModel.from_pretrained(model_name, output_attentions=True)
    model.to(device)
    model.eval()

    # 2. 图像预处理
    if not os.path.exists(args.image_path):
        print(f"Error: Image path {args.image_path} not found.")
        sys.exit(1)

    with open(args.image_path, 'rb') as f:
        img_pil = Image.open(f).convert('RGB')

    # 使用处理器进行标准的 resize 和 normalize
    inputs = processor(images=img_pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 3. 模型前向传播
    with torch.no_grad():
        outputs = model(**inputs)
    
    # outputs.attentions 是一个元组，包含每一层的注意力 tensor
    # 每一个 tensor 形状为: [Batch_Size, Num_Heads, Seq_Len, Seq_Len]
    # Seq_Len = 1 (CLS) + 196 (Patches) = 197
    all_attentions = outputs.attentions

    # 检查层数索引是否有效
    if args.layer < 0 or args.layer >= len(all_attentions):
        print(f"Error: Layer index {args.layer} is out of range. Model has {len(all_attentions)} layers.")
        sys.exit(1)

    # 4. 获取指定层的注意力
    # att_mat: [1, 12, 197, 197]
    att_mat = all_attentions[args.layer]
    
    # 获取 Patch 数量的宽高 (224/16 = 14)
    patch_size = 16
    w_featmap = inputs['pixel_values'].shape[2] // patch_size
    h_featmap = inputs['pixel_values'].shape[3] // patch_size
    
    # 我们只关注 [CLS] token (index 0) 对所有图片 Patch (index 1 to end) 的注意力
    # 形状变换: [1, 12, 197, 197] -> [12, 196]
    attentions = att_mat[0, :, 0, 1:]

    nh = attentions.shape[0] # number of heads (12)

    # 5. 阈值处理与生成 Mask
    th_attn = None
    if args.threshold is not None:
        # 为了计算阈值，我们保留原始 flat 形状
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - args.threshold)
        
        # 恢复原来的顺序
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
            
        # Reshape 成 2D 特征图: [12, 14, 14]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        
        # 插值放大到原图尺寸 (224x224)
        th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

    # 6. 处理用于显示的 Heatmap
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    # 插值放大
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

    # 7. 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存原始图片 (反归一化以便显示)
    # ViTImageProcessor 的均值和方差
    mean = torch.tensor(processor.image_mean).view(3, 1, 1).to(device)
    std = torch.tensor(processor.image_std).view(3, 1, 1).to(device)
    img_tensor = inputs['pixel_values'][0] * std + mean
    img_tensor = torch.clamp(img_tensor, 0, 1)
    
    torchvision.utils.save_image(img_tensor, os.path.join(args.output_dir, "img.png"))
    
    # 读取保存的原图用于 OpenCV/Matplotlib 绘制
    image_vis = skimage.io.imread(os.path.join(args.output_dir, "img.png"))

    print(f"Saving visualizations for Layer {args.layer} to {args.output_dir}...")
    
    for j in range(nh):
        # 保存热力图 (Attention Map)
        fname = os.path.join(args.output_dir, "attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format='png')
        
        # 保存带 Mask 的图
        if args.threshold is not None:
            display_instances(image_vis, th_attn[j], fname=os.path.join(args.output_dir, "mask_head" + str(j) + ".png"), blur=False)
            
    combine_images(args.output_dir, output_filename=os.path.join(args.output_dir, f"combined_layer_{args.layer}.png"))
    print("Done.")
