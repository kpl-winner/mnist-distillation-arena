"""
Probability Smoothing（spec 第四十五节最小示例）

将概率与均匀分布混合：
    tilde_p = (1 - alpha) * p_v + alpha * uniform
通常不改变 Top-1 标签（LabelErrorRate ≈ 0），但降低置信度信息。
"""

from __future__ import annotations

import numpy as np

from .base import BaseDefense, register_defense


@register_defense("smoothing")
class SmoothingDefense(BaseDefense):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.alpha = float(cfg.get("defense", {}).get("smoothing_alpha", 0.10))

    def defend(self, ctx, query):
        probs = np.asarray(query["raw_probabilities"], dtype=np.float64).copy()
        uniform = np.ones_like(probs) / self.num_classes
        probs = (1.0 - self.alpha) * probs + self.alpha * uniform
        probs = self._renormalize(probs)
        labels = np.argmax(probs, axis=1)
        return {"labels": labels, "probabilities": probs}
