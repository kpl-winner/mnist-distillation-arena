"""
MNIST 黑盒模型蒸馏攻防对抗赛 - 主入口

子命令:
    download    下载 MNIST 数据集到本地
    train       训练 Victim Model
    eval        运行完整 Attack × Defense 评测矩阵（自动化验证）
    match       运行单组 Attack × Defense 对阵
    list        列出已注册的攻击 / 防御方法

示例:
    python main.py download
    python main.py train --quick
    python main.py eval --quick
    python main.py match --attack soft_label --defense noise --quick
    python main.py match --use-submission --quick
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from judge.config_utils import apply_quick, ensure_utf8_stdout, load_config

ensure_utf8_stdout()

_ROOT = os.path.dirname(os.path.abspath(__file__))
_CFG_PATH = os.path.join(_ROOT, "config.yaml")


def _load(args):
    cfg = load_config(_CFG_PATH)
    if getattr(args, "quick", False):
        cfg = apply_quick(cfg)
    return cfg


def cmd_download(args):
    from judge.dataset import ensure_mnist, load_mnist
    cfg = _load(args)
    ensure_mnist(cfg["data"]["root"])
    load_mnist(cfg["data"]["root"])


def cmd_train(args):
    from judge.dataset import build_splits
    from judge.label_mapping import apply_mapping, generate_permutation
    from judge.train_victim import save_victim, train_victim
    cfg = _load(args)
    device = cfg.get("device", "cpu")
    splits = build_splits(cfg, seed=args.perm_seed)
    mapping = generate_permutation(args.perm_seed)
    vy_perm = apply_mapping(splits.victim_train_y, mapping)
    victim, acc = train_victim(cfg, splits.victim_train_x, vy_perm, device, seed=args.train_seed)
    path = os.path.join(cfg["victim"]["checkpoint_dir"],
                        f"victim_p{args.perm_seed}_t{args.train_seed}.pt")
    save_victim(victim, acc, mapping, cfg, path)
    print(f"最终 acc={acc:.4f}（阈值 {cfg['victim']['acc_threshold']}）")


def cmd_eval(args):
    from judge.evaluator import Evaluator
    cfg = _load(args)
    device = cfg.get("device", "cpu")
    attacks = args.attacks.split(",") if args.attacks else None
    defenses = args.defenses.split(",") if args.defenses else None
    ev = Evaluator(cfg, device=device)
    ev.run_full_evaluation(
        attack_names=attacks, defense_names=defenses,
        num_permutations=args.num_perm, num_seeds=args.num_seeds,
    )


def cmd_match(args):
    # 复用 scripts/run_match.py 的逻辑
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_match", os.path.join(_ROOT, "scripts", "run_match.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # 直接调用其 main 会重新解析 argv，这里手动设置 sys.argv
    sys.argv = ["run_match"] + _build_match_argv(args)
    mod.main()


def _build_match_argv(args):
    out = []
    if args.attack:
        out += ["--attack", args.attack]
    if args.defense:
        out += ["--defense", args.defense]
    out += ["--perm-seed", str(args.perm_seed), "--train-seed", str(args.train_seed)]
    if args.quick:
        out.append("--quick")
    if args.use_submission:
        out.append("--use-submission")
    return out


def cmd_list(args):
    from attacks import list_attacks
    from defenses import list_defenses
    print("已注册攻击方法:", list_attacks())
    print("已注册防御方法:", list_defenses())


def main():
    ap = argparse.ArgumentParser(description="MNIST 黑盒模型蒸馏攻防对抗赛")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download", help="下载 MNIST 到本地")
    p_dl.add_argument("--quick", action="store_true")
    p_dl.set_defaults(func=cmd_download)

    p_tr = sub.add_parser("train", help="训练 Victim Model")
    p_tr.add_argument("--perm-seed", type=int, default=42)
    p_tr.add_argument("--train-seed", type=int, default=42)
    p_tr.add_argument("--quick", action="store_true")
    p_tr.set_defaults(func=cmd_train)

    p_ev = sub.add_parser("eval", help="运行完整 Attack×Defense 评测矩阵")
    p_ev.add_argument("--attacks", type=str, default=None)
    p_ev.add_argument("--defenses", type=str, default=None)
    p_ev.add_argument("--num-perm", type=int, default=None)
    p_ev.add_argument("--num-seeds", type=int, default=None)
    p_ev.add_argument("--quick", action="store_true")
    p_ev.set_defaults(func=cmd_eval)

    p_mt = sub.add_parser("match", help="运行单组 Attack×Defense 对阵")
    p_mt.add_argument("--attack", default="soft_label")
    p_mt.add_argument("--defense", default="smoothing")
    p_mt.add_argument("--perm-seed", type=int, default=42)
    p_mt.add_argument("--train-seed", type=int, default=42)
    p_mt.add_argument("--quick", action="store_true")
    p_mt.add_argument("--use-submission", action="store_true")
    p_mt.set_defaults(func=cmd_match)

    p_ls = sub.add_parser("list", help="列出已注册方法")
    p_ls.set_defaults(func=cmd_list)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
