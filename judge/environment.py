"""
environment.py
==============
裁判环境 DistillationEnv —— 攻防解耦的核心桥梁。

攻击方仅通过 `env.query(images)` 与 `env.create_student_model()` 访问目标模型；
防御方仅在 `defend(ctx, query)` 中收到 (图像, 原始标签, 原始概率) 及受限上下文。

内部流程（spec 第四十九节 query_api.py）：
    Query Images -> Victim Model -> Raw Label/Prob -> Defense -> Protected Output

环境同时统计：
    Query Count / Label Error Rate / Probability Distortion（供裁判计分）
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import to_tensor
from .student_model import build_student
from .validator import (
    BudgetExceededError,
    IllegalDefenseOutputError,
    validate_defense_output,
)


class DefenseContext:
    """传递给防御方的受限环境视图。

    仅暴露查询状态与历史，**不**暴露 victim 模型、原始概率、query() 方法
    或攻击方训练状态，从而严格满足 spec 第十节威胁模型。
    """

    def __init__(self, env: "DistillationEnv"):
        self._env = env

    @property
    def query_count(self) -> int:
        return self._env.query_count

    @property
    def query_budget(self) -> int:
        return self._env.query_budget

    @property
    def num_classes(self) -> int:
        return self._env.num_classes

    @property
    def image_shape(self):
        return self._env.image_shape

    @property
    def history(self):
        """本会话中防御方已产出的 (labels, probabilities) 列表。"""
        return self._env._defended_history  # noqa: SLF001


class DistillationEnv:
    def __init__(
        self,
        victim: torch.nn.Module,
        defense,
        query_pool_x: np.ndarray,
        query_budget: int,
        cfg: dict,
        device: str = "cpu",
        num_classes: int = 10,
    ):
        self.victim = victim.to(device).eval()
        self.defense = defense
        self.query_pool = query_pool_x            # [N,28,28] uint8（攻击方可访问）
        self.query_budget = int(query_budget)
        self.query_count = 0
        self.cfg = cfg
        self.device = device
        self.num_classes = num_classes
        self.image_shape = (1, 28, 28)

        # 受限上下文（传给 defense.defend）
        self._ctx = DefenseContext(self)

        # 裁判内部记录（不暴露给攻击方 / 防御方）
        self._raw_labels_log: list[np.ndarray] = []
        self._raw_probs_log: list[np.ndarray] = []
        self._defended_labels_log: list[np.ndarray] = []
        self._defended_probs_log: list[np.ndarray] = []
        self._defended_history: list[tuple[np.ndarray, np.ndarray]] = []

        # 会话开始：重置防御状态
        self.defense.reset()

    # ------------------------------------------------------------------ #
    # 攻击方接口
    # ------------------------------------------------------------------ #
    def create_student_model(self) -> torch.nn.Module:
        """返回一个全新的统一 Student Model（spec 第九节）。"""
        return build_student(self.cfg["student"]["arch"], self.num_classes).to(self.device)

    def query(self, images) -> dict:
        """查询目标模型，返回 {"labels": [N], "probabilities": [N,10]}。

        一次 Batch Query 中包含多少图片，就计为多少次查询（spec 第十一节）。
        """
        images = np.asarray(images)
        n = images.shape[0]
        if n == 0:
            return {"labels": np.empty(0, dtype=np.int64),
                    "probabilities": np.empty((0, self.num_classes), dtype=np.float32)}

        if self.query_count + n > self.query_budget:
            raise BudgetExceededError(
                f"查询超预算: 已用 {self.query_count} + 本次 {n} > 预算 {self.query_budget}"
            )

        # 1) Victim 推理
        raw_probs = self._victim_predict(images)
        raw_labels = np.argmax(raw_probs, axis=1).astype(np.int64)

        # 2) 防御处理
        query = {
            "images": images,
            "raw_labels": raw_labels,
            "raw_probabilities": raw_probs,
        }
        defended = self.defense.defend(self._ctx, query)

        # 3) 合法性校验
        validate_defense_output(defended, raw_labels, raw_probs, self.num_classes)

        defended_labels = np.asarray(defended["labels"], dtype=np.int64).copy()
        defended_probs = np.asarray(defended["probabilities"], dtype=np.float64).copy()

        # 4) 强约束：tilde_y = argmax(tilde_p)（spec 第十六节）
        defended_labels = np.argmax(defended_probs, axis=1).astype(np.int64)

        # 5) 记录
        self._raw_labels_log.append(raw_labels)
        self._raw_probs_log.append(raw_probs)
        self._defended_labels_log.append(defended_labels)
        self._defended_probs_log.append(defended_probs)
        self._defended_history.append((defended_labels, defended_probs))
        self.query_count += n

        return {"labels": defended_labels, "probabilities": defended_probs}

    # ------------------------------------------------------------------ #
    # 内部：Victim 推理
    # ------------------------------------------------------------------ #
    def _victim_predict(self, images: np.ndarray) -> np.ndarray:
        x = to_tensor(images, self.cfg).astype(np.float32)   # [N,1,28,28]
        t = torch.from_numpy(x).to(self.device)
        with torch.no_grad():
            logits = self.victim(t)
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy().astype(np.float64)

    # ------------------------------------------------------------------ #
    # 裁判接口：会话级防御服务指标
    # ------------------------------------------------------------------ #
    def defense_session_metrics(self) -> dict:
        """聚合本会话所有查询的防御服务指标（spec 第十五/三十节）。"""
        if not self._raw_labels_log:
            return {"label_error_rate": 0.0, "probability_retention": 1.0,
                    "service_quality": 1.0, "valid": True, "num_queries": 0}
        raw_labels = np.concatenate(self._raw_labels_log)
        raw_probs = np.concatenate(self._raw_probs_log)
        def_labels = np.concatenate(self._defended_labels_log)
        def_probs = np.concatenate(self._defended_probs_log)

        from . import metrics as M
        ler = M.label_error_rate(def_labels, raw_labels)
        lr = M.label_retention(ler)
        pr = M.probability_retention(raw_probs, def_probs)
        return {
            "label_error_rate": ler,
            "label_retention": lr,
            "probability_retention": pr,
            "service_quality": M.service_quality(lr, pr),
            "valid": ler <= self.cfg["evaluation"]["defense_label_error_limit"] + 1e-9,
            "num_queries": int(len(raw_labels)),
        }

    @property
    def budget_exhausted(self) -> bool:
        return self.query_count >= self.query_budget
