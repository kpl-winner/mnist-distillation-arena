"""
Random Noise Defense（利用标签错误率预算的较强基线）

策略：
  1. 对所有查询的概率叠加小幅高斯噪声（不改变 Top-1，破坏软目标精度）；
  2. 在 10% 标签错误预算内（目标 9%，留安全余量），将部分样本的
     概率最大类替换为错误类，从而向攻击方注入错误监督信号。

会话级预算跟踪保证 LabelErrorRate <= 10%（spec 第十五节硬性约束）。
"""

from __future__ import annotations

import numpy as np

from .base import BaseDefense, register_defense


@register_defense("noise")
class NoiseDefense(BaseDefense):
    def __init__(self, cfg):
        super().__init__(cfg)
        dcfg = cfg.get("defense", {})
        self.sigma = float(dcfg.get("noise_sigma", 0.05))
        self.target_rate = float(dcfg.get("noise_target_rate", 0.09))
        self._total = 0
        self._flipped = 0

    def reset(self):
        super().reset()
        self._total = 0
        self._flipped = 0

    def defend(self, ctx, query):
        raw_labels = np.asarray(query["raw_labels"]).copy()
        probs = np.asarray(query["raw_probabilities"], dtype=np.float64).copy()
        n = len(raw_labels)

        # 会话级可翻转配额
        total_after = self._total + n
        max_flips = int(np.floor(self.target_rate * total_after))
        flips_available = max(0, max_flips - self._flipped)

        flip_mask = np.zeros(n, dtype=bool)
        if flips_available > 0:
            k = min(flips_available, n)
            chosen = self._rng.choice(n, size=k, replace=False)
            flip_mask[chosen] = True

        for i in range(n):
            true_c = int(raw_labels[i])
            if flip_mask[i]:
                # 选一个错误类并抬升为 argmax
                wrongs = [c for c in range(self.num_classes) if c != true_c]
                wrong = int(self._rng.choice(wrongs))
                mx = probs[i].max()
                probs[i, wrong] = mx + 0.02
            else:
                # 小幅噪声，但保持原 Top-1
                noise = self._rng.normal(0.0, self.sigma, size=probs.shape[1])
                probs[i] = np.clip(probs[i] + noise, 1e-6, None)
                if int(np.argmax(probs[i])) != true_c:
                    probs[i, true_c] = probs[i].max() + 1e-3
            probs[i] /= probs[i].sum()

        self._total += n
        self._flipped += int(flip_mask.sum())

        labels = np.argmax(probs, axis=1).astype(np.int64)
        return {"labels": labels, "probabilities": probs}
