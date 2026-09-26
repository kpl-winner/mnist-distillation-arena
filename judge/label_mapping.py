"""
label_mapping.py
================
随机标签置换 π: Y -> Y（spec 第五节）。

- 每轮生成一个一一映射（随机排列）。
- 类别一致性：同一原始类别的图像统一映射到同一新标签。
- label_mapping 只保存在裁判环境中，不暴露给攻击方 / 防御方。
"""

from __future__ import annotations

import numpy as np


def generate_permutation(seed: int, num_classes: int = 10) -> np.ndarray:
    """生成随机排列 π，返回长度为 num_classes 的数组。

    mapping[i] = π(i)，即原始类别 i 映射到 mapping[i]。
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_classes)
    return perm.astype(np.int64)


def apply_mapping(labels: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    """对标签数组执行置换：y' = π(y)。"""
    return mapping[labels]


def inverse_mapping(mapping: np.ndarray) -> np.ndarray:
    """返回逆映射 π^{-1}。"""
    inv = np.empty_like(mapping)
    inv[mapping] = np.arange(len(mapping))
    return inv
