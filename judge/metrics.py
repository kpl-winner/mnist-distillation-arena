"""
metrics.py
==========
全部评测指标实现（spec 第二十五 ~ 三十三节）。

  LabelAgreement        第 25 节
  ProbabilityAgreement  第 26 节  (1 - JSD/ln2)
  AttackScore           第 27 节  (0.8*LA + 0.2*PA)
  AccStudent            第 28 节
  LabelErrorRate        第 15/29 节
  LabelRetention        第 29 节
  TVD / ProbabilityRetention  第 30 节
  ServiceQuality        第 31 节  (0.7*LR + 0.3*PR)
  HardSuppression / SoftSuppression / DistillationSuppression  第 32 节
  DefenseScore          第 33 节  (0.75*DS + 0.25*SQ)
"""

from __future__ import annotations

import math

import numpy as np

_LN2 = math.log(2.0)
_EPS = 1e-12
_RANDOM_LEVEL = 0.1   # 十分类随机预测水平


# --------------------------------------------------------------------------- #
# 基础距离 / 一致性
# --------------------------------------------------------------------------- #
def label_agreement(y_student: np.ndarray, y_victim: np.ndarray) -> float:
    """标签行为一致率：mean(y_s == y_v)。"""
    y_student = np.asarray(y_student).ravel()
    y_victim = np.asarray(y_victim).ravel()
    if len(y_student) != len(y_victim):
        raise ValueError("长度不一致")
    return float(np.mean(y_student == y_victim))


def _xlogy(x, y):
    """x * log(y)，约定 0*log(0)=0。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    out = np.zeros_like(x)
    mask = x > _EPS
    out[mask] = x[mask] * np.log(np.clip(y[mask], _EPS, None))
    return out


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """JSD(p, q)，自然对数，返回每条样本的 JSD（形状 [N]）。

    0 <= JSD <= ln2。
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / np.clip(p.sum(axis=1, keepdims=True), _EPS, None)
    q = q / np.clip(q.sum(axis=1, keepdims=True), _EPS, None)
    m = 0.5 * (p + q)
    jsd = 0.5 * np.sum(_xlogy(p, p / m) + _xlogy(q, q / m), axis=1)
    return np.clip(jsd, 0.0, _LN2)


def probability_agreement(p_victim: np.ndarray, p_student: np.ndarray) -> float:
    """概率行为一致率：1 - mean(JSD)/ln2，范围 [0,1]。"""
    jsd = jensen_shannon_divergence(p_victim, p_student)
    return float(1.0 - np.mean(jsd) / _LN2)


def attack_score(label_agr: float, prob_agr: float) -> float:
    """攻击方蒸馏得分：0.8*LA + 0.2*PA。"""
    return 0.8 * label_agr + 0.2 * prob_agr


def student_accuracy(y_student: np.ndarray, y_true_perm: np.ndarray) -> float:
    """Student 在随机标签任务上的分类准确率（spec 第 28 节）。"""
    return label_agreement(y_student, y_true_perm)


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """TVD = 0.5 * sum |p - q|，返回每条样本（形状 [N]）。"""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return 0.5 * np.sum(np.abs(p - q), axis=1)


# --------------------------------------------------------------------------- #
# 防御服务指标
# --------------------------------------------------------------------------- #
def label_error_rate(defended_labels: np.ndarray, raw_labels: np.ndarray) -> float:
    """LabelErrorRate = mean(tilde_y != y_v)。"""
    return float(np.mean(np.asarray(defended_labels).ravel()
                         != np.asarray(raw_labels).ravel()))


def label_retention(label_error: float) -> float:
    """LabelRetention = 1 - LabelErrorRate。"""
    return 1.0 - label_error


def probability_retention(p_victim: np.ndarray, p_defended: np.ndarray) -> float:
    """ProbabilityRetention = 1 - mean(TVD)。"""
    tvd = total_variation_distance(p_victim, p_defended)
    return float(1.0 - np.mean(tvd))


def service_quality(label_ret: float, prob_ret: float) -> float:
    """ServiceQuality = 0.7*LabelRetention + 0.3*ProbabilityRetention。"""
    return 0.7 * label_ret + 0.3 * prob_ret


# --------------------------------------------------------------------------- #
# 防御蒸馏抑制
# --------------------------------------------------------------------------- #
def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


def hard_suppression(la_no_def: float, la_with_def: float) -> float:
    """HardSuppression = Clip((LA0 - LAD)/(LA0 - 0.1), 0, 1)。"""
    denom = la_no_def - _RANDOM_LEVEL
    if denom <= 0:
        return 0.0
    return _clip((la_no_def - la_with_def) / denom)


def soft_suppression(pa_no_def: float, pa_with_def: float) -> float:
    """SoftSuppression = Clip((PA0 - PAD)/PA0, 0, 1)。"""
    if pa_no_def <= 0:
        return 0.0
    return _clip((pa_no_def - pa_with_def) / pa_no_def)


def distillation_suppression(hard_sup: float, soft_sup: float) -> float:
    """DistillationSuppression = 0.8*Hard + 0.2*Soft。"""
    return 0.8 * hard_sup + 0.2 * soft_sup


def defense_score(distill_sup: float, service_q: float,
                  label_error: float, error_limit: float = 0.10) -> float:
    """DefenseScore = 0.75*DistillationSuppression + 0.25*ServiceQuality。

    硬性约束：LabelErrorRate > 限制 -> DefenseScore = 0。
    """
    if label_error > error_limit + 1e-9:
        return 0.0
    return 0.75 * distill_sup + 0.25 * service_q


# --------------------------------------------------------------------------- #
# 汇总：单次攻击 vs 单次防御的完整指标
# --------------------------------------------------------------------------- #
def compute_attack_metrics(
    y_student: np.ndarray,
    y_victim: np.ndarray,
    p_student: np.ndarray,
    p_victim: np.ndarray,
    y_true_perm: np.ndarray | None = None,
) -> dict:
    """计算 Victim/Student 行为一致性相关指标。"""
    la = label_agreement(y_student, y_victim)
    pa = probability_agreement(p_victim, p_student)
    out = {
        "label_agreement": la,
        "probability_agreement": pa,
        "attack_score": attack_score(la, pa),
    }
    if y_true_perm is not None:
        out["student_accuracy"] = student_accuracy(y_student, y_true_perm)
    return out


def compute_defense_metrics(
    defended_labels: np.ndarray,
    raw_labels: np.ndarray,
    defended_probs: np.ndarray,
    raw_probs: np.ndarray,
) -> dict:
    """计算防御服务质量指标。"""
    ler = label_error_rate(defended_labels, raw_labels)
    lr = label_retention(ler)
    pr = probability_retention(raw_probs, defended_probs)
    return {
        "label_error_rate": ler,
        "label_retention": lr,
        "probability_retention": pr,
        "service_quality": service_quality(lr, pr),
        "valid": ler <= 0.10 + 1e-9,
    }
