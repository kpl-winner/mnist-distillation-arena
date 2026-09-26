"""
victim_model.py
===============
基础目标模型 (Victim Model)。

公开阶段采用小型 CNN（spec 第七节）：
    Conv3x3 -> ReLU -> MaxPool
    Conv3x3 -> ReLU -> MaxPool
    Conv3x3 -> ReLU
    FC -> 10

通过 registry 可扩展多种 Victim 变体（spec 第四十一节 Victim-A/B/C）。
"""

from __future__ import annotations

import torch
import torch.nn as nn

_VICTIM_REGISTRY: dict[str, type[nn.Module]] = {}


def register_victim(name: str):
    def deco(cls):
        _VICTIM_REGISTRY[name] = cls
        cls.victim_name = name
        return cls
    return deco


@register_victim("victim_cnn")
class VictimCNN(nn.Module):
    """spec 第七节默认目标模型。"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),   # 28x28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                  # 14x14
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                  # 7x7
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


@register_victim("lenet")
class LeNet(nn.Module):
    """经典 LeNet-5 风格变体（用于隐藏阶段 Victim-A）。"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),    # 28x28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                  # 14x14
            nn.Conv2d(6, 16, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                  # 5x5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_victim(arch: str = "victim_cnn", num_classes: int = 10) -> nn.Module:
    if arch not in _VICTIM_REGISTRY:
        raise KeyError(f"未知 victim 架构: {arch}（可用: {list(_VICTIM_REGISTRY)}）")
    return _VICTIM_REGISTRY[arch](num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
