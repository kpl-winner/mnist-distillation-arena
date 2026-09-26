"""攻击方法包。导入即注册全部内置攻击。"""

from .base import BaseAttack, get_attack, list_attacks, register_attack  # noqa: F401
from . import hard_label, soft_label, active_query  # noqa: F401
