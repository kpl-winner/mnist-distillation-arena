"""
training_utils.py
=================
训练 / 推理复用工具（不依赖 torchvision）。

  - random_affine       离线数据增强（攻击方可用于查询图像，spec 第十节允许）
  - distillation_loss   KL 蒸馏损失（带温度）
  - predict             模型批量推理 -> (probabilities, labels)
  - images_to_tensor    [N,28,28] uint8 -> [N,1,28,28] 标准化张量
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import to_tensor


def images_to_tensor(images: np.ndarray, cfg: dict, device: str = "cpu") -> torch.Tensor:
    return torch.from_numpy(to_tensor(images, cfg).astype(np.float32)).to(device)


# --------------------------------------------------------------------------- #
# 数据增强：随机仿射（平移 + 旋转 + 缩放），torchvision-free
# --------------------------------------------------------------------------- #
def random_affine(
    images: torch.Tensor,
    max_translate: float = 0.12,
    max_angle: float = 15.0,
    max_scale: float = 0.10,
) -> torch.Tensor:
    """images: [B,1,28,28]，返回增强后同形状张量。"""
    if images.dim() != 4:
        raise ValueError("期望 [B,1,28,28]")
    B = images.shape[0]
    device = images.device
    angle = (torch.rand(B, device=device) * 2 - 1) * math.radians(max_angle)
    scale = 1.0 + (torch.rand(B, device=device) * 2 - 1) * max_scale
    tx = (torch.rand(B, device=device) * 2 - 1) * max_translate
    ty = (torch.rand(B, device=device) * 2 - 1) * max_translate

    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    # 仿射矩阵 [B,2,3]：先缩放旋转，再平移
    theta = torch.stack([
        torch.stack([scale * cos_a, -scale * sin_a, tx], dim=1),
        torch.stack([scale * sin_a,  scale * cos_a, ty], dim=1),
    ], dim=1)
    grid = F.affine_grid(theta, images.size(), align_corners=False)
    return F.grid_sample(images, grid, align_corners=False, padding_mode="zeros")


# --------------------------------------------------------------------------- #
# 蒸馏损失
# --------------------------------------------------------------------------- #
def distillation_loss(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    temperature: float = 4.0,
) -> torch.Tensor:
    """KL(teacher || softmax(student/T)) * T^2。

    teacher_probs 为 API 返回的（可能被防御扰动过的）概率分布。
    """
    T = temperature
    log_s = F.log_softmax(student_logits / T, dim=1)
    p_t = teacher_probs.clamp(min=1e-8)
    log_t = torch.log(p_t)
    # KL(p_t || p_s) = sum p_t (log p_t - log p_s)
    kl = (p_t * (log_t - log_s)).sum(dim=1).mean() * (T * T)
    return kl


# --------------------------------------------------------------------------- #
# 推理
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(
    model: torch.nn.Module,
    images: np.ndarray,
    cfg: dict,
    device: str = "cpu",
    batch_size: int = 512,
):
    """返回 (probabilities[N,C] float64, labels[N] int64)。"""
    model.eval()
    probs_chunks = []
    x = to_tensor(images, cfg).astype(np.float32)
    n = x.shape[0]
    for s in range(0, n, batch_size):
        t = torch.from_numpy(x[s:s + batch_size]).to(device)
        logits = model(t)
        probs_chunks.append(F.softmax(logits, dim=1).cpu().numpy())
    probs = np.concatenate(probs_chunks, axis=0).astype(np.float64)
    labels = np.argmax(probs, axis=1).astype(np.int64)
    return probs, labels
