"""
submission/attack.py
====================
攻击方提交接口（spec 第十九节）。

  def attack(env, task) -> {"student_state_dict": ...}

本文件从 config.yaml 的 `submission.attack` 读取要调用的攻击方法，
并委托给 attacks/ 注册表中的实现。参赛者可：
  - 修改 config.yaml 选择不同基线；
  - 或直接替换本函数体内的逻辑。
"""

from __future__ import annotations

import os
import sys

# 将项目根目录加入 sys.path，使 submission 可独立运行
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from attacks import get_attack                       # noqa: E402
from judge.config_utils import load_config            # noqa: E402

_CFG = load_config(os.path.join(_ROOT, "config.yaml"))
_ATTACK_NAME = _CFG.get("submission", {}).get("attack", "soft_label")
_ATTACK = get_attack(_ATTACK_NAME, _CFG)


def attack(env, task: dict) -> dict:
    """
    输入：
        env: 裁判环境，可使用 env.query() / env.create_student_model()
        task: {"query_images", "query_budget", "num_classes", "image_shape"}

    输出：
        {"student_state_dict": dict}
    """
    return _ATTACK.run(env, task)
