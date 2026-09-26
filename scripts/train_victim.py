"""独立训练 Victim Model 并保存 checkpoint（供评测复用）。

用法:
    python scripts/train_victim.py [--perm-seed 42] [--train-seed 42] [--quick]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judge.config_utils import apply_quick, ensure_utf8_stdout, load_config

ensure_utf8_stdout()
from judge.dataset import build_splits
from judge.label_mapping import apply_mapping, generate_permutation
from judge.train_victim import save_victim, train_victim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm-seed", type=int, default=42)
    ap.add_argument("--train-seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config(os.path.join(root, "config.yaml"))
    if args.quick:
        cfg = apply_quick(cfg)
    device = cfg.get("device", "cpu")

    splits = build_splits(cfg, seed=args.perm_seed)
    mapping = generate_permutation(args.perm_seed)
    vy_perm = apply_mapping(splits.victim_train_y, mapping)

    victim, acc = train_victim(cfg, splits.victim_train_x, vy_perm, device, seed=args.train_seed)
    path = os.path.join(cfg["victim"]["checkpoint_dir"],
                        f"victim_p{args.perm_seed}_t{args.train_seed}.pt")
    save_victim(victim, acc, mapping, cfg, path)
    print(f"[train_victim] 最终 acc={acc:.4f}  阈值={cfg['victim']['acc_threshold']}")


if __name__ == "__main__":
    main()
