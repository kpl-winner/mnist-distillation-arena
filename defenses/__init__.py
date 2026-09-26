"""防御方法包。导入即注册全部内置防御。"""

from .base import BaseDefense, get_defense, list_defenses, register_defense  # noqa: F401
from . import none, rounding, smoothing, sharpening, noise  # noqa: F401
