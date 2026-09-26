"""
defenses/base.py
===============
防御方法基类与注册表（与攻击完全解耦）。

防御方部署在 Victim API 输出端，仅看到 (图像, 原始标签, 原始概率) 及受限上下文，
产出 (defended_label, defended_probability)。环境随后强制 tilde_y = argmax(tilde_p)。

实现一个新防御只需：
    @register_defense("my_defense")
    class MyDefense(BaseDefense):
        def defend(self, ctx, query): ...
        # 有跨查询状态时重写 reset()
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

_DEFENSE_REGISTRY: dict[str, type["BaseDefense"]] = {}


def register_defense(name: str):
    def deco(cls):
        cls.name = name
        _DEFENSE_REGISTRY[name] = cls
        return cls
    return deco


def get_defense(name: str, cfg: dict) -> "BaseDefense":
    if name not in _DEFENSE_REGISTRY:
        raise KeyError(f"未知防御方法: {name}（可用: {list(_DEFENSE_REGISTRY)}）")
    return _DEFENSE_REGISTRY[name](cfg)


def list_defenses() -> list[str]:
    return sorted(_DEFENSE_REGISTRY)


class BaseDefense(ABC):
    """所有防御方法的基类。"""

    name: str = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.num_classes = 10
        self._seed = int(cfg.get("seed", 42))
        self._rng = np.random.default_rng(self._seed + self._stable_offset())
        self.reset()

    @abstractmethod
    def defend(self, ctx, query: dict) -> dict:
        """输入 query={images, raw_labels, raw_probabilities}，
        返回 {"labels": ndarray, "probabilities": ndarray}。

        注意：返回的 labels 会被环境强制改为 argmax(probabilities)（spec 第十六节），
        因此防御方若要改变预测标签，必须让对应类别在 probabilities 中成为最大值。
        """
        raise NotImplementedError

    def reset(self) -> None:
        """每个攻击会话开始时调用，重置跨查询状态与随机流。"""
        self._rng = np.random.default_rng(self._seed + self._stable_offset())

    # ------------------------------------------------------------------ #
    # 共享工具
    # ------------------------------------------------------------------ #
    def _stable_offset(self) -> int:
        """基于类名的确定性偏移（避免 hash() 跨运行不稳定）。"""
        return sum((i + 1) * ord(c) for i, c in enumerate(self.name))

    @staticmethod
    def _renormalize(probs: np.ndarray) -> np.ndarray:
        probs = np.clip(probs, 0.0, None)
        s = probs.sum(axis=1, keepdims=True)
        return probs / np.clip(s, 1e-12, None)


class FunctionDefenseAdapter(BaseDefense):
    """将 spec 风格的 `defend(env, query)` 函数适配为 BaseDefense 对象。

    供 spec 风格裁判加载 submission/defense.py 后接入 DistillationEnv 使用。
    """

    name = "function_adapter"

    def __init__(self, defend_fn, cfg: dict):
        super().__init__(cfg)
        self._fn = defend_fn

    def defend(self, ctx, query: dict) -> dict:
        return self._fn(ctx, query)
