"""
attacks/base.py
===============
攻击方法基类与注册表（与防御完全解耦）。

攻击方仅通过 `env.query()` 获得被防御扰动后的 (label, probability)，
据此训练统一 Student Model。攻击方不知道当前激活的防御算法。

实现一个新攻击只需：
    @register_attack("my_attack")
    class MyAttack(BaseAttack):
        def query_strategy(self, env, task): ...
        # 可选：重写 train_student 自定义蒸馏损失
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from judge.dataset import to_tensor
from judge.training_utils import random_affine

_ATTACK_REGISTRY: dict[str, type["BaseAttack"]] = {}


def register_attack(name: str):
    def deco(cls):
        cls.name = name
        _ATTACK_REGISTRY[name] = cls
        return cls
    return deco


def get_attack(name: str, cfg: dict) -> "BaseAttack":
    if name not in _ATTACK_REGISTRY:
        raise KeyError(f"未知攻击方法: {name}（可用: {list(_ATTACK_REGISTRY)}）")
    return _ATTACK_REGISTRY[name](cfg)


def list_attacks() -> list[str]:
    return sorted(_ATTACK_REGISTRY)


class BaseAttack(ABC):
    """所有攻击方法的基类。"""

    name: str = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = cfg.get("device", "cpu")
        self.scfg = cfg["student"]

    # ------------------------------------------------------------------ #
    # 主流程
    # ------------------------------------------------------------------ #
    def run(self, env, task: dict) -> dict:
        """默认流程：查询 -> 训练 -> 返回 state_dict。"""
        images, labels, probs = self.query_strategy(env, task)
        student = self.train_student(env, images, labels, probs)
        return {"student_state_dict": student.state_dict()}

    @abstractmethod
    def query_strategy(self, env, task: dict):
        """选择查询图片并调用 env.query()。

        返回 (queried_images[N,28,28] uint8, labels[N] int, probs[N,10] float)。
        """
        raise NotImplementedError

    def train_student(self, env, images, labels, probs):
        """默认使用 soft-label 蒸馏；子类可重写。"""
        return self._train_distill(env, images, labels, probs,
                                   soft_w=1.0, hard_w=0.0)

    # ------------------------------------------------------------------ #
    # 共享训练工具
    # ------------------------------------------------------------------ #
    def _train_distill(
        self, env, images, labels, probs,
        soft_w: float = 1.0, hard_w: float = 0.0,
        epochs: int | None = None, augment: bool = True,
    ):
        """统一学生模型蒸馏训练。

        loss = soft_w * KL(teacher_prob || softmax(z_s/T)) * T^2
             + hard_w * CE(label)
        每个 epoch 对查询图像做随机仿射增强（spec 第十节允许）。
        """
        scfg = self.scfg
        epochs = epochs if epochs is not None else scfg["epochs"]
        T = scfg.get("temperature", 4.0)
        bs = scfg["batch_size"]
        lr = scfg["lr"]
        wd = scfg["weight_decay"]

        student = env.create_student_model()
        optimizer = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=wd)

        x = to_tensor(images, self.cfg).astype(np.float32)              # [N,1,28,28]
        y = np.asarray(labels, dtype=np.int64)
        p = np.asarray(probs, dtype=np.float32)
        ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(p))
        loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=False)

        n = len(x)
        for _ in range(epochs):
            student.train()
            total = 0.0
            for xb, yb, pb in loader:
                xb, yb, pb = xb.to(self.device), yb.to(self.device), pb.to(self.device)
                if augment and n > 0:
                    xb = random_affine(xb)
                optimizer.zero_grad()
                logits = student(xb)
                loss = torch.tensor(0.0, device=self.device)
                if soft_w > 0:
                    log_s = F.log_softmax(logits / T, dim=1)
                    p_t = pb.clamp(min=1e-8)
                    log_t = torch.log(p_t)
                    kl = (p_t * (log_t - log_s)).sum(dim=1).mean() * (T * T)
                    loss = loss + soft_w * kl
                if hard_w > 0:
                    loss = loss + hard_w * F.cross_entropy(logits, yb)
                loss.backward()
                optimizer.step()
                total += loss.item() * xb.size(0)
        return student

    # ------------------------------------------------------------------ #
    # 查询辅助
    # ------------------------------------------------------------------ #
    @staticmethod
    def _select_indices(pool_size: int, n: int, rng: np.random.Generator) -> np.ndarray:
        n = min(n, pool_size)
        return rng.choice(pool_size, size=n, replace=False)
