import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
from skimage.measure import find_contours
import skimage.io
from transformers import ViTImageProcessor, ViTForImageClassification
crate_alpha_pretrain_path = '/home/hpo/Documents/pretrained_models/crate/crate_alpha_B16.pth'

crate_alpha_pretrain_path = "/home/ckbu/CRATE-GCD/cubc.pt"
# --------------------------------------------------------
# 1. 你的 CRATE 模型定义 (保持不变)
# --------------------------------------------------------
from einops import rearrange, repeat
import torch.nn.functional as F
def combine_images(source_folder, output_filename="combined_result.png"):
    # 1. 准备文件列表和对应的标签
    # 列表包含元组: (文件名, 底部显示的英文标签)
    image_list = []
    
    # 添加原始图片
    image_list.append(("img.png", "Original Image"))
    
    # 添加12张注意力头图片 (attn-head0.png 到 attn-head11.png)
    for i in range(12):
        filename = f"attn-head{i}.png"
        label = f"Head {i}"
        image_list.append((filename, label))
    
    total_images = len(image_list)  # 应该是 13 张

    # 2. 创建画布
    # figsize=(宽, 高)，宽度设置大一点以容纳13张图片
    fig, axes = plt.subplots(1, total_images, figsize=(24, 3))
    
    # 3. 遍历并绘图
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
            print(f"警告: 文件不存在 -> {path}")
            ax.text(0.5, 0.5, "Missing", ha='center', va='center')
        
        # 4. 设置样式和标签
        ax.set_xlabel(label, fontsize=12, fontweight='bold') # 标签写在下面
        ax.set_xticks([]) # 移除X轴刻度
        ax.set_yticks([]) # 移除Y轴刻度
        
        # 移除边框（可选，如果想要纯白背景）
        for spine in ax.spines.values():
            spine.set_visible(False)

    # 5. 调整布局并保存
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"成功！图片已保存为: {output_filename}")
    plt.close()

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class OvercompleteISTABlock(nn.Module):
    def __init__(self, d, overcomplete_ratio=4, eta=0.1, lmbda=0.1):
        super(OvercompleteISTABlock, self).__init__()
        self.eta = eta
        self.lmbda = lmbda
        self.overcomplete_ratio = overcomplete_ratio
        self.d = d
        self.D = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d, overcomplete_ratio * d)))
        self.D1 = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d, overcomplete_ratio * d)))

    def forward(self, x):
        negative_lasso_grad = torch.einsum("pd,nlp->nld", self.D, x)
        z1 = F.relu(self.eta * negative_lasso_grad - self.eta * self.lmbda)
        Dz1 = torch.einsum("dp,nlp->nld", self.D, z1)
        lasso_grad = torch.einsum("pd,nlp->nld", self.D, Dz1 - x)
        z2 = F.relu(z1 - self.eta * lasso_grad - self.eta * self.lmbda)
        xhat = torch.einsum("dp,nlp->nld", self.D1, z2)
        return xhat

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.qkv = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h = self.heads)
        dots = torch.matmul(w, w.transpose(-1, -2)) * self.scale
        attn = self.attend(dots) # <--- 我们需要捕获这里的输出
        attn = self.dropout(attn)
        out = torch.matmul(attn, w)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.heads = heads
        self.depth = depth
        self.dim = dim
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, OvercompleteISTABlock(d = dim))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x 

    def forward_features(self, x, nth_layers):
        output = []
        for i, (attn, ff) in enumerate(self.layers):
            x = attn(x) + x
            x = ff(x) + x
            if len(self.layers) - i in nth_layers:
                output.append(x[:, 0])
        return output 

class CRATE(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=dim, kernel_size=patch_size, stride=patch_size, bias=True, padding='valid')
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, dropout)
        self.to_latent = nn.Identity()

    def forward(self, img):
        x = self.conv1(img)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.transformer(x)
        x = x[:, 0]
        x = self.to_latent(x)
        return x

def CRATE_base():
    return CRATE(image_size=224, patch_size=16, dim=768, depth=12, heads=12, dropout=0.0, emb_dropout=0.0, dim_head=768//12)

# --------------------------------------------------------
# 2. 辅助函数 (可视化相关)
# --------------------------------------------------------

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
    print(f"{fname} saved.")
    plt.close(fig)

# --------------------------------------------------------
# 3. 主程序
# --------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize CRATE Self-Attention maps')
    parser.add_argument("--image_path", default=None, type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(224, 224), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='crate_vis_output', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=0.6, help="Percentage of mass to keep.")
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--layer', default=16, type=int, help='Patch resolution of the model.')
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 构建模型
    print("Building CRATE_base model...")
    model = CRATE_base()
    
    # --- 修正后的加载逻辑 ---
    print(f"Loading weights from: {crate_alpha_pretrain_path}")
    checkpoint = torch.load(crate_alpha_pretrain_path, map_location='cpu')
    
    # 1. 处理字典嵌套 (取出 state_dict)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # 2. 修正键名 (去除 "0." 前缀)
    new_state_dict = {}
    for k, v in state_dict.items():
        # 如果是 cubc 模型那种带 "0." 前缀的
        if k.startswith('0.'):
            new_key = k[2:] # 去掉前两个字符 "0."
            new_state_dict[new_key] = v
        # 如果是带 "model." 前缀的 (另一种常见情况)
        elif k.startswith('model.'):
            new_key = k[6:]
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v

    # 3. 加载权重
    # strict=False 依然保留，因为 Model 2 确实少了 mlp_head，但现在主干会被正确加载
    msg = model.load_state_dict(new_state_dict, strict=False)
    
    print(f"加载报告 - 缺失的键: {len(msg.missing_keys)} 个")
    print(f"加载报告 - 多余的键: {len(msg.unexpected_keys)} 个")
    
    # 检查核心层是否加载成功 (如果 transformer.layers.0 没有加载，肯定有问题)
    if any("transformer.layers.0" in k for k in msg.missing_keys):
        print("【严重警告】主干网络未加载成功！请检查键名匹配！")
    else:
        print("【成功】主干网络权重加载成功。")

    model.to(device)
    model.eval()

    # model = CRATE_base()
    # state_dict = torch.load(crate_alpha_pretrain_path, map_location='cpu')
    # model.load_state_dict(state_dict, strict=False)

    # model.to(device)


    # model.eval()

    # --- 关键：使用 Hook 获取 Attention ---
    # 我们需要捕获 model.transformer.layers 中最后一层的 Attention 里的 Softmax 输出
    # 结构: Transformer -> layers (list) -> [0] (Attention Block) -> [0] (PreNorm) -> fn (Attention) -> attend (Softmax)
    
    attn_outputs = []
    def hook_fn(module, input, output):
        # output shape: [batch, heads, N, N]
        attn_outputs.append(output.detach().cpu())

    # 注册 hook 到最后一层
    # layers[-1] 是最后一个 block (包含 Attn 和 ISTA)
    # layers[-1][0] 是 PreNorm(Attention)
    # layers[-1][0].fn 是 Attention 实例
    # layers[-1][0].fn.attend 是 Softmax 层
    target_layer = model.transformer.layers[args.layer][0].fn.attend
    hook_handle = target_layer.register_forward_hook(hook_fn)

    with open(args.image_path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')

    transform = pth_transforms.Compose([
        pth_transforms.Resize(args.image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img_tensor = transform(img)

    w, h = img_tensor.shape[1] - img_tensor.shape[1] % args.patch_size, img_tensor.shape[2] - img_tensor.shape[2] % args.patch_size
    img_tensor = img_tensor[:, :w, :h].unsqueeze(0).to(device)

    with torch.no_grad():
        model(img_tensor)
    
    # 移除 hook
    hook_handle.remove()

    if len(attn_outputs) == 0:
        print("Error: No attention captured.")
        sys.exit(1)

    # 获取注意力图 [Batch, Heads, N, N]
    # N = 1 (CLS) + (H*W)/P^2
    attentions = attn_outputs[0] 
    nh = attentions.shape[1] # number of heads

    # 获取特征图尺寸
    w_featmap = img_tensor.shape[-2] // args.patch_size
    h_featmap = img_tensor.shape[-1] // args.patch_size

    # 提取 [CLS] token 对所有 patch 的注意力
    # index 0 is CLS, so we look at row 0, and columns 1 to end (skipping CLS itself)
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    # 阈值处理
    th_attn = None
    if args.threshold is not None:
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - args.threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        # 插值放大到原图尺寸
        th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

    # 处理原始注意力图用于热力图显示
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存原图
    torchvision.utils.save_image(torchvision.utils.make_grid(img_tensor, normalize=True, scale_each=True), os.path.join(args.output_dir, "img.png"))
    
    # 读取保存的原图用于 OpenCV/Matplotlib 绘制
    image_vis = skimage.io.imread(os.path.join(args.output_dir, "img.png"))

    print(f"Saving visualizations to {args.output_dir}...")
    
    for j in range(nh):
        # 保存热力图
        fname = os.path.join(args.output_dir, "attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format='png')
        
        # 保存带 mask 的图
        if args.threshold is not None:
            display_instances(image_vis, th_attn[j], fname=os.path.join(args.output_dir, "mask_head" + str(j) + ".png"), blur=False)
    combine_images(args.output_dir,output_filename=os.path.join(args.output_dir,"combined_result_layer"+str(args.layer)+".png"))
    print("Done.")
