"""
submission/defense.py
=====================
防御方提交接口（spec 第二十三节）。

  def defend(env, query) -> {"labels": ..., "probabilities": ...}

从 config.yaml 的 `submission.defense` 读取要调用的防御方法，
委托给 defenses/ 注册表中的实现。参赛者可：
  - 修改 config.yaml 选择不同基线；
  - 或直接替换本函数体内的逻辑。

注意：env 为受限上下文（DefenseContext），仅暴露 query_count / query_budget
/ num_classes / image_shape / history，不暴露 Victim 模型或原始概率。
"""

from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from defenses import get_defense                      # noqa: E402
from judge.config_utils import load_config            # noqa: E402

_CFG = load_config(os.path.join(_ROOT, "config.yaml"))
_DEFENSE_NAME = _CFG.get("submission", {}).get("defense", "smoothing")
_DEFENSE = get_defense(_DEFENSE_NAME, _CFG)


def defend(env, query: dict) -> dict:
    """
    输入：
        query = {"images", "raw_labels", "raw_probabilities"}
    输出：
        {"labels": ndarray, "probabilities": ndarray}
    """
    return _DEFENSE.defend(env, query)
