"""
student_model.py
================
统一学生模型 (Student Model)。

- 所有攻击方共享同一结构，仅比较查询策略 / 蒸馏方法 / 损失函数（spec 第九节）。
- 参数量 <= 1M。
- 裁判环境通过 `env.create_student_model()` 提供实例。
"""

from __future__ import annotations

import torch
import torch.nn as nn

_STUDENT_REGISTRY: dict[str, type[nn.Module]] = {}


def register_student(name: str):
    def deco(cls):
        _STUDENT_REGISTRY[name] = cls
        cls.student_name = name
        return cls
    return deco


@register_student("student_cnn")
class StudentCNN(nn.Module):
    """统一学生模型，参数量约 0.1M（远小于 1M 限制）。"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),   # 28x28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                  # 14x14
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                  # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_student(arch: str = "student_cnn", num_classes: int = 10) -> nn.Module:
    if arch not in _STUDENT_REGISTRY:
        raise KeyError(f"未知 student 架构: {arch}（可用: {list(_STUDENT_REGISTRY)}）")
    model = _STUDENT_REGISTRY[arch](num_classes=num_classes)
    # 合法性自检：参数量 <= 1M
    n = count_parameters(model)
    if n > 1_000_000:
        raise RuntimeError(f"Student 参数量 {n} 超过 1M 限制")
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
