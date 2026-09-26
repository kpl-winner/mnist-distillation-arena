"""
train_victim.py
===============
训练基础目标模型 (Victim Model)。

- 使用标签置换后的训练集 D_train' = {(x_i, π(y_i))}（spec 第七节）。
- 必须达到 Acc_victim >= 阈值，否则重新训练（spec 第八节）。
- 训练结果保存为 checkpoint，供评测复用。
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .dataset import to_tensor
from .victim_model import build_victim


def _make_loaders(victim_x, victim_y_perm, cfg, device):
    """构造训练 / 验证 DataLoader。从 victim_train 中切出 10% 作为验证集。"""
    n = len(victim_x)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_val = max(1, n // 10)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    x = to_tensor(victim_x, cfg).astype(np.float32)
    y = victim_y_perm.astype(np.int64)

    train_ds = TensorDataset(
        torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx]))
    val_ds = TensorDataset(
        torch.from_numpy(x[val_idx]), torch.from_numpy(y[val_idx]))

    bs = cfg["victim"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    return train_loader, val_loader


def train_victim_once(cfg, victim_x, victim_y_perm, device="cpu", seed=42):
    """训练一次 Victim Model，返回 (model, val_acc)。"""
    vcfg = cfg["victim"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_victim(vcfg["arch"], num_classes=10).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=vcfg["lr"], weight_decay=vcfg["weight_decay"])

    train_loader, val_loader = _make_loaders(victim_x, victim_y_perm, cfg, device)
    epochs = vcfg["epochs"]

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        val_acc = _evaluate_accuracy(model, val_loader, device)
        print(f"[victim] epoch {ep+1}/{epochs}  loss={total_loss/len(train_loader.dataset):.4f}  val_acc={val_acc:.4f}")
    return model, val_acc


@torch.no_grad()
def _evaluate_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return correct / max(total, 1)


def train_victim(cfg, victim_x, victim_y_perm, device="cpu", seed=42):
    """训练 Victim，若未达准确率阈值则重新训练（spec 第八节）。

    返回 (model, acc)。
    """
    vcfg = cfg["victim"]
    threshold = vcfg["acc_threshold"]
    max_retrains = vcfg.get("max_retrains", 3)

    best_model, best_acc = None, -1.0
    for attempt in range(1, max_retrains + 1):
        print(f"[victim] 训练尝试 {attempt}/{max_retrains}（种子 {seed}）")
        model, acc = train_victim_once(cfg, victim_x, victim_y_perm, device, seed)
        if acc > best_acc:
            best_model, best_acc = model, acc
        if acc >= threshold:
            print(f"[victim] 达到阈值 {threshold:.2f}: acc={acc:.4f}")
            return best_model, best_acc
        seed += 1   # 换种子重训
    print(f"[victim] 警告: 未达阈值 {threshold:.2f}，使用最佳 acc={best_acc:.4f}")
    return best_model, best_acc


# --------------------------------------------------------------------------- #
# checkpoint 持久化
# --------------------------------------------------------------------------- #
def save_victim(model, acc, mapping, cfg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "acc": acc,
        "mapping": mapping,
        "arch": cfg["victim"]["arch"],
        "victim_train_size": int(cfg["data"]["splits"]["victim_train"]),
        "epochs": int(cfg["victim"]["epochs"]),
    }, path)
    print(f"[victim] 已保存 checkpoint -> {path} (acc={acc:.4f})")


def load_victim(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_victim(ckpt["arch"], num_classes=10).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["acc"], ckpt["mapping"]


def victim_checkpoint_matches(ckpt: dict, mapping, cfg: dict, victim_train_size: int) -> bool:
    """检查 checkpoint 是否与当前配置兼容（标签置换 / 训练规模 / 轮数）。"""
    if not np.array_equal(ckpt.get("mapping"), mapping):
        return False
    if ckpt.get("victim_train_size") != victim_train_size:
        return False
    if ckpt.get("epochs") != cfg["victim"]["epochs"]:
        return False
    return True
