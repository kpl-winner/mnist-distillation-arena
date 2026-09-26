"""
validator.py
============
合法性检查（spec 第二十节）。

  - 查询预算            N_query <= QueryBudget
  - Student 参数合法    state_dict 可加载到统一 Student Model
  - Defense 概率合法    形状 / 非负 / 和为 1 / 有限
  - Defense 标签错误率  <= 10%（在 environment 会话级聚合中判定）
"""

from __future__ import annotations

import numpy as np
import torch


class BudgetExceededError(Exception):
    """攻击方查询次数超过预算。"""


class IllegalDefenseOutputError(Exception):
    """防御方输出不合法（形状 / 概率约束）。"""


class IllegalAttackOutputError(Exception):
    """攻击方输出不合法（state_dict 无法加载）。"""


# --------------------------------------------------------------------------- #
# 防御输出
# --------------------------------------------------------------------------- #
def validate_defense_output(
    defended: dict,
    raw_labels: np.ndarray,
    raw_probs: np.ndarray,
    num_classes: int,
) -> None:
    if not isinstance(defended, dict) or "labels" not in defended or "probabilities" not in defended:
        raise IllegalDefenseOutputError("防御输出必须包含 'labels' 与 'probabilities'")

    labels = np.asarray(defended["labels"])
    probs = np.asarray(defended["probabilities"], dtype=np.float64)

    if labels.shape != raw_labels.shape:
        raise IllegalDefenseOutputError(
            f"labels 形状不匹配: {labels.shape} != {raw_labels.shape}"
        )
    if probs.shape != raw_probs.shape:
        raise IllegalDefenseOutputError(
            f"probabilities 形状不匹配: {probs.shape} != {raw_probs.shape}"
        )
    if not np.all(np.isfinite(probs)):
        raise IllegalDefenseOutputError("probabilities 含非有限值")
    if np.any(probs < -1e-9):
        raise IllegalDefenseOutputError("probabilities 存在负值")
    probs = np.clip(probs, 0.0, None)
    row_sums = probs.sum(axis=1)
    if np.any(np.abs(row_sums - 1.0) > 1e-4):
        raise IllegalDefenseOutputError(
            f"probabilities 行和不为 1（最大偏差 {np.max(np.abs(row_sums-1.0)):.2e}）"
        )
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise IllegalDefenseOutputError(f"labels 超出 [0,{num_classes}) 范围")


# --------------------------------------------------------------------------- #
# 攻击输出
# --------------------------------------------------------------------------- #
def validate_attack_result(result: dict, student: torch.nn.Module) -> None:
    if not isinstance(result, dict) or "student_state_dict" not in result:
        raise IllegalAttackOutputError("攻击输出必须包含 'student_state_dict'")
    sd = result["student_state_dict"]
    try:
        student.load_state_dict(sd)
    except Exception as e:  # noqa: BLE001
        raise IllegalAttackOutputError(f"student_state_dict 加载失败: {e}") from e


def check_query_budget(used: int, budget: int) -> None:
    if used > budget:
        raise BudgetExceededError(f"查询次数 {used} 超过预算 {budget}")
