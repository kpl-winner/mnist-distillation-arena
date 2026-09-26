"""
A0-1: Hard Label Distillation（spec 第四十二节）

攻击方只使用 API 返回的预测标签构造 (image, label)，
以 CrossEntropy 训练 Student Model。
"""

from __future__ import annotations

import numpy as np

from .base import BaseAttack, register_attack


@register_attack("hard_label")
class HardLabelAttack(BaseAttack):
    """仅用硬标签蒸馏。"""

    def query_strategy(self, env, task):
        pool = task["query_images"]
        budget = task["query_budget"]
        rng = np.random.default_rng(12345)

        n = min(budget, len(pool))
        idx = self._select_indices(len(pool), n, rng)
        images = pool[idx]

        # 分批查询，避免单次显存/内存过大
        labels, probs = self._batched_query(env, images, batch=256)
        return images, labels, probs

    def train_student(self, env, images, labels, probs):
        # 仅使用硬标签 CE（忽略概率），仍保留数据增强
        return self._train_distill(env, images, labels, probs,
                                   soft_w=0.0, hard_w=1.0)

    @staticmethod
    def _batched_query(env, images, batch=256):
        all_l, all_p = [], []
        for s in range(0, len(images), batch):
            r = env.query(images[s:s + batch])
            all_l.append(r["labels"])
            all_p.append(r["probabilities"])
        return (np.concatenate(all_l), np.concatenate(all_p, axis=0))
