"""
Soft Label Distillation

攻击方使用 API 返回的完整概率分布作为软目标，
以 KL 散度蒸馏 Student Model。同时叠加少量硬标签 CE 提升稳定性。
"""

from __future__ import annotations

import numpy as np

from .base import BaseAttack, register_attack


@register_attack("soft_label")
class SoftLabelAttack(BaseAttack):
    """软标签（概率）蒸馏。"""

    def query_strategy(self, env, task):
        pool = task["query_images"]
        budget = task["query_budget"]
        rng = np.random.default_rng(54321)

        n = min(budget, len(pool))
        idx = self._select_indices(len(pool), n, rng)
        images = pool[idx]

        labels, probs = self._batched_query(env, images, batch=256)
        return images, labels, probs

    def train_student(self, env, images, labels, probs):
        # 主项 KL 软蒸馏 + 小权重硬标签稳定
        return self._train_distill(env, images, labels, probs,
                                   soft_w=1.0, hard_w=0.3)

    @staticmethod
    def _batched_query(env, images, batch=256):
        all_l, all_p = [], []
        for s in range(0, len(images), batch):
            r = env.query(images[s:s + batch])
            all_l.append(r["labels"])
            all_p.append(r["probabilities"])
        return (np.concatenate(all_l), np.concatenate(all_p, axis=0))
