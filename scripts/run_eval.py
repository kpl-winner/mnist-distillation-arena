"""自动化验证：运行完整 Attack × Defense 全对阵评测矩阵。

用法:
    python scripts/run_eval.py [--quick] [--attacks a,b] [--defenses d,e]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judge.config_utils import apply_quick, ensure_utf8_stdout, load_config

ensure_utf8_stdout()
from judge.evaluator import Evaluator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="缩小规模以快速跑通")
    ap.add_argument("--attacks", type=str, default=None, help="逗号分隔的攻击方法名")
    ap.add_argument("--defenses", type=str, default=None, help="逗号分隔的防御方法名")
    ap.add_argument("--num-perm", type=int, default=None)
    ap.add_argument("--num-seeds", type=int, default=None)
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config(os.path.join(root, "config.yaml"))
    if args.quick:
        cfg = apply_quick(cfg)
    device = cfg.get("device", "cpu")

    attacks = args.attacks.split(",") if args.attacks else None
    defenses = args.defenses.split(",") if args.defenses else None

    ev = Evaluator(cfg, device=device)
    ev.run_full_evaluation(
        attack_names=attacks, defense_names=defenses,
        num_permutations=args.num_perm, num_seeds=args.num_seeds,
    )


if __name__ == "__main__":
    main()
