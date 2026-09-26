"""
dataset.py
==========
手写数字数据集加载与划分。

- 不依赖 torchvision，直接解析 MNIST 原始 IDX 文件（CPU 环境更稳健）。
- 自动从公网镜像下载到本地 `data/` 目录。
- 按 spec 第六节划分为：
      Victim Training Set / Public Query Pool / Hidden Test Set

公开接口：
    ensure_mnist(data_dir)      下载并缓存 MNIST
    load_mnist(data_dir)        返回 (train_x, train_y, test_x, test_y)
    build_splits(cfg, seed)     返回三个划分 (含原始标签，未经标签置换)
"""

from __future__ import annotations

import gzip
import os
import struct
import time
from dataclasses import dataclass

import numpy as np
import requests

# 使用 PyTorch 官方维护的 S3 镜像（原 yann.lecun.com 经常不可达）
_MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
# 各文件期望字节数（用于校验下载完整性，避免残缺文件被误判为已完成）
_MNIST_EXPECTED_SIZE = {
    "train_images": 9912422,
    "train_labels": 28881,
    "test_images": 1648877,
    "test_labels": 4542,
}
_MNIST_MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]


# --------------------------------------------------------------------------- #
# 下载
# --------------------------------------------------------------------------- #
def _download_file(url: str, dst: str) -> None:
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    done = 0
    with open(dst, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            if chunk:
                f.write(chunk)
                done += len(chunk)
    if total and done != total:
        raise IOError(f"下载不完整: {url} (期望 {total}, 实际 {done})")


def ensure_mnist(data_dir: str) -> None:
    """若本地不存在 MNIST 原始文件（或大小不符），则从镜像下载。"""
    os.makedirs(data_dir, exist_ok=True)
    for key, fname in _MNIST_FILES.items():
        dst = os.path.join(data_dir, fname)
        expected = _MNIST_EXPECTED_SIZE[key]
        if os.path.exists(dst) and os.path.getsize(dst) == expected:
            continue
        if os.path.exists(dst):
            print(f"[dataset] {fname} 大小不符（{os.path.getsize(dst)}/{expected}），重新下载")
            os.remove(dst)
        ok = False
        for mirror in _MNIST_MIRRORS:
            url = mirror + fname
            try:
                print(f"[dataset] 下载 {fname} <- {url}")
                _download_file(url, dst)
                if os.path.getsize(dst) != expected:
                    print(f"[dataset] 大小校验失败: {os.path.getsize(dst)}/{expected}")
                    os.remove(dst)
                    continue
                ok = True
                break
            except Exception as e:  # noqa: BLE001
                print(f"[dataset] 失败: {e}")
                if os.path.exists(dst):
                    os.remove(dst)
        if not ok:
            raise RuntimeError(f"无法下载 {fname}，请检查网络或手动放置到 {data_dir}")


# --------------------------------------------------------------------------- #
# IDX 解析
# --------------------------------------------------------------------------- #
def _read_idx_images(path: str) -> np.ndarray:
    """读取 IDX3 图像文件，返回 [N,28,28] uint8。"""
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 0x00000803:
            raise ValueError(f"非 IDX3 图像文件: {path} magic={magic:#x}")
        buf = f.read(num * rows * cols)
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(num, rows, cols)
    return arr


def _read_idx_labels(path: str) -> np.ndarray:
    """读取 IDX1 标签文件，返回 [N] uint8。"""
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 0x00000801:
            raise ValueError(f"非 IDX1 标签文件: {path} magic={magic:#x}")
        buf = f.read(num)
        arr = np.frombuffer(buf, dtype=np.uint8)
    return arr


def load_mnist(data_dir: str):
    """加载 MNIST，返回 (train_x, train_y, test_x, test_y)。

    图像为 [N,28,28] uint8，标签为 [N] uint8。
    结果缓存为 .npz 以加速后续加载。
    """
    ensure_mnist(data_dir)
    cache = os.path.join(data_dir, "mnist_cache.npz")
    if os.path.exists(cache):
        data = np.load(cache)
        return data["train_x"], data["train_y"], data["test_x"], data["test_y"]

    t0 = time.time()
    train_x = _read_idx_images(os.path.join(data_dir, _MNIST_FILES["train_images"]))
    train_y = _read_idx_labels(os.path.join(data_dir, _MNIST_FILES["train_labels"]))
    test_x = _read_idx_images(os.path.join(data_dir, _MNIST_FILES["test_images"]))
    test_y = _read_idx_labels(os.path.join(data_dir, _MNIST_FILES["test_labels"]))
    np.savez(cache, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
    print(f"[dataset] 加载完成: train={train_x.shape} test={test_x.shape} ({time.time()-t0:.1f}s)")
    return train_x, train_y, test_x, test_y


# --------------------------------------------------------------------------- #
# 划分
# --------------------------------------------------------------------------- #
@dataclass
class DataSplits:
    """三个数据划分。图像为 [N,28,28] uint8，标签为原始标签（未经置换）。"""

    victim_train_x: np.ndarray
    victim_train_y: np.ndarray
    query_pool_x: np.ndarray       # 攻击方可访问的图像（无标签对外）
    query_pool_y: np.ndarray       # 真实标签（仅裁判持有，不暴露给攻击方）
    hidden_test_x: np.ndarray
    hidden_test_y: np.ndarray

    @property
    def num_classes(self) -> int:
        return 10

    @property
    def image_shape(self):
        return (1, 28, 28)


def build_splits(cfg: dict, seed: int = 42) -> DataSplits:
    """加载 MNIST 并按配置规模随机划分为三份。

    划分方式：将 train(60k)+test(10k) 合并后打乱，再按配置切分，
    确保 Victim Training / Query Pool / Hidden Test 三者互不相交。
    """
    data_cfg = cfg["data"]
    root = data_cfg["root"]
    sizes = data_cfg["splits"]

    train_x, train_y, test_x, test_y = load_mnist(root)
    all_x = np.concatenate([train_x, test_x], axis=0)
    all_y = np.concatenate([train_y, test_y], axis=0)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(all_x))
    all_x = all_x[perm]
    all_y = all_y[perm]

    n_v = sizes["victim_train"]
    n_q = sizes["query_pool"]
    n_t = sizes["hidden_test"]
    total_needed = n_v + n_q + n_t
    if total_needed > len(all_x):
        raise RuntimeError(
            f"数据不足以满足划分需求: 需要 {total_needed}, 实际 {len(all_x)}"
        )

    s = 0
    vx, vy = all_x[s:s + n_v], all_y[s:s + n_v]; s += n_v
    qx, qy = all_x[s:s + n_q], all_y[s:s + n_q]; s += n_q
    tx, ty = all_x[s:s + n_t], all_y[s:s + n_t]; s += n_t

    return DataSplits(
        victim_train_x=vx, victim_train_y=vy,
        query_pool_x=qx, query_pool_y=qy,
        hidden_test_x=tx, hidden_test_y=ty,
    )


# --------------------------------------------------------------------------- #
# 图像预处理工具（供模型训练 / 推理复用）
# --------------------------------------------------------------------------- #
def to_tensor(images: np.ndarray, cfg: dict) -> np.ndarray:
    """[N,28,28] uint8 -> [N,1,28,28] float32 标准化。"""
    mean = cfg["data"]["normalize"]["mean"]
    std = cfg["data"]["normalize"]["std"]
    x = images.astype(np.float32) / 255.0
    x = (x - mean) / std
    return x[:, None, :, :]  # [N,1,28,28]
