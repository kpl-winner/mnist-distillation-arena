"""
Active Query Distillation（主动学习查询策略）

流程：
  1. 随机查询少量种子样本；
  2. 用已查询数据训练临时 student；
  3. 对未查询池计算 student 预测熵，挑选最不确定的样本查询；
  4. 重复直到预算耗尽；
  5. 用全部已查询数据做最终蒸馏训练。

在有限预算下通常优于随机查询。
"""

from __future__ import annotations

import numpy as np

from .base import BaseAttack, register_attack
from judge.training_utils import predict


@register_attack("active_query")
class ActiveQueryAttack(BaseAttack):
    """基于不确定性的主动查询 + 软标签蒸馏。"""

    def query_strategy(self, env, task):
        pool = task["query_images"]
        budget = task["query_budget"]
        rng = np.random.default_rng(98765)

        n_pool = len(pool)
        queried_mask = np.zeros(n_pool, dtype=bool)

        # 初始种子查询
        init_n = min(max(1, budget // 8), 100)
        seed_idx = rng.choice(n_pool, size=init_n, replace=False)
        collected_idx = list(seed_idx)
        queried_mask[seed_idx] = True

        images, labels, probs = self._query_indices(env, pool, collected_idx)

        # 迭代主动查询
        round_idx = 0
        while len(collected_idx) < budget:
            round_idx += 1
            remaining = budget - len(collected_idx)
            batch = min(self._round_batch(budget), remaining)

            # 训练临时 student
            prov = self._train_distill(
                env, images, labels, probs,
                soft_w=1.0, hard_w=0.2,
                epochs=max(3, self.scfg["epochs"] // 2),
                augment=True,
            )

            # 在未查询池上预测，计算熵
            unq_idx = np.where(~queried_mask)[0]
            if len(unq_idx) == 0:
                break
            # 为加速，仅在一个子集上评分（池可能很大）
            eval_idx = unq_idx if len(unq_idx) <= 4096 else rng.choice(
                unq_idx, size=4096, replace=False)
            p_unq, _ = predict(prov, pool[eval_idx], self.cfg, self.device, batch_size=512)
            entropy = -(p_unq * np.log(p_unq + 1e-12)).sum(axis=1)

            # 混合选择：一半按不确定性，一半随机（避免纯 top-entropy 选中离群点）
            n_uncertain = max(1, batch // 2)
            n_random = batch - n_uncertain
            top = np.argsort(-entropy)[:n_uncertain]
            uncertain_pick = eval_idx[top]
            # 随机部分从未查询且未被本轮选中中取
            chosen_set = set(int(i) for i in uncertain_pick)
            avail = np.array([i for i in eval_idx if int(i) not in chosen_set])
            if len(avail) >= n_random:
                rand_pick = rng.choice(avail, size=n_random, replace=False)
            else:
                rand_pick = avail
            pick = [int(i) for i in np.concatenate([uncertain_pick, rand_pick])
                    if not queried_mask[int(i)]]
            if not pick:
                break

            # 查询并合并
            q_imgs = pool[pick]
            r = env.query(q_imgs)
            images = np.concatenate([images, q_imgs], axis=0)
            labels = np.concatenate([labels, r["labels"]])
            probs = np.concatenate([probs, r["probabilities"]], axis=0)
            collected_idx.extend(pick)
            queried_mask[pick] = True

        return images, labels, probs

    def train_student(self, env, images, labels, probs):
        # 主动查询已收集数据，做最终完整蒸馏
        return self._train_distill(env, images, labels, probs,
                                   soft_w=1.0, hard_w=0.2)

    @staticmethod
    def _round_batch(budget: int) -> int:
        return max(1, budget // 8)

    @staticmethod
    def _query_indices(env, pool, indices):
        idx = list(indices)
        images = pool[idx]
        r = env.query(images)
        return images, r["labels"], r["probabilities"]
