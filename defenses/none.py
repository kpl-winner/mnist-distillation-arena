"""
D0-1: No Defense（spec 第四十三节）

直接返回 raw_label / raw_probabilities，用于建立攻击基准。
"""

from __future__ import annotations

import numpy as np

from .base import BaseDefense, register_defense


@register_defense("none")
class NoDefense(BaseDefense):
    def defend(self, ctx, query):
        return {
            "labels": np.asarray(query["raw_labels"]).copy(),
            "probabilities": np.asarray(query["raw_probabilities"], dtype=np.float64).copy(),
        }
