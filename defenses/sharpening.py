"""
Probability Sharpening（spec 第四十八节基线）

对概率做温度锐化：
    tilde_p = softmax(log(p) / T),  T < 1  ->  更尖锐（接近 one-hot）

作为防御通常较弱：更干净的软目标反而有利于攻击方蒸馏，
但会改变概率分布形状（影响 ProbabilityAgreement）。保留作为对照基线。
"""

from __future__ import annotations

import numpy as np

from .base import BaseDefense, register_defense


@register_defense("sharpening")
class SharpeningDefense(BaseDefense):
    def __init__(self, cfg):
        super().__init__(cfg)
        # T < 1 锐化；越大越平滑
        self.T = float(cfg.get("defense", {}).get("sharpening_T", 0.5))

    def defend(self, ctx, query):
        probs = np.asarray(query["raw_probabilities"], dtype=np.float64).copy()
        probs = np.clip(probs, 1e-8, 1.0)
        # 等价于 p**(1/T) 后归一化
        sharpened = probs ** (1.0 / self.T)
        sharpened = self._renormalize(sharpened)
        labels = np.argmax(sharpened, axis=1)
        return {"labels": labels, "probabilities": sharpened}
