"""下载 MNIST 到本地 data/ 目录并构建 .npz 缓存。"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judge.config_utils import ensure_utf8_stdout, load_config

ensure_utf8_stdout()
from judge.dataset import ensure_mnist, load_mnist


def main():
    cfg = load_config(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"))
    root = cfg["data"]["root"]
    print(f"[download] 目标目录: {os.path.abspath(root)}")
    ensure_mnist(root)
    train_x, _, test_x, _ = load_mnist(root)
    print(f"[download] 完成: train={train_x.shape} test={test_x.shape}")


if __name__ == "__main__":
    main()
