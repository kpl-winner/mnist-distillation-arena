"""
D0-2: Probability Rounding（spec 第四十三节）

降低概率输出精度（保留 k 位小数）后重新归一化。
通常不改变 Top-1 标签，但削弱软目标的细粒度信息。
"""

from __future__ import annotations

import numpy as np

from .base import BaseDefense, register_defense


@register_defense("rounding")
class RoundingDefense(BaseDefense):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.decimals = int(cfg.get("defense", {}).get("rounding_decimals", 2))

    def defend(self, ctx, query):
        probs = np.asarray(query["raw_probabilities"], dtype=np.float64).copy()
        probs = np.round(probs, self.decimals)
        probs = self._renormalize(probs)
        labels = np.argmax(probs, axis=1)
        return {"labels": labels, "probabilities": probs}
