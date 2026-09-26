"""
config_utils.py
===============
配置加载与快速模式覆盖。
"""

from __future__ import annotations

import copy
import os

import yaml


def load_config(path: str = "config.yaml") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_quick(cfg: dict) -> dict:
    """快速模式：缩小数据规模与训练轮数，便于在 CPU 上快速跑通全流程。

    不改变评测逻辑，仅缩减规模。
    """
    cfg = copy.deepcopy(cfg)
    cfg["data"]["splits"] = {"victim_train": 6000, "query_pool": 3000, "hidden_test": 2000}
    cfg["victim"]["epochs"] = 4
    cfg["student"]["epochs"] = 6
    cfg["query"]["budget"] = 300
    cfg["evaluation"]["eval_subset"] = 2000
    return cfg


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_utf8_stdout() -> None:
    """在 Windows GBK 控制台下避免非 ASCII 输出崩溃（中文/符号以 UTF-8 输出）。"""
    import sys
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
