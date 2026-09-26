"""运行单组 Attack × Defense 对阵，输出详细指标（含无防御基线对比）。

用法:
    python scripts/run_match.py --attack soft_label --defense noise [--quick]
    python scripts/run_match.py --use-submission          # 走 submission/ 接口
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judge.config_utils import apply_quick, ensure_utf8_stdout, load_config

ensure_utf8_stdout()
from judge.evaluator import Evaluator
from judge.environment import DistillationEnv
from judge.training_utils import predict
from judge.validator import validate_attack_result
from judge import metrics as M
from attacks import get_attack
from defenses import get_defense
from defenses.base import FunctionDefenseAdapter


def _run(attack_obj_or_fn, defense, round_ctx, cfg, device, is_fn_attack=False):
    env = DistillationEnv(
        round_ctx["victim"], defense, round_ctx["splits"].query_pool_x,
        cfg["query"]["budget"], cfg, device, num_classes=10,
    )
    task = {
        "query_images": round_ctx["splits"].query_pool_x,
        "query_budget": cfg["query"]["budget"],
        "num_classes": 10, "image_shape": (1, 28, 28),
    }
    if is_fn_attack:
        result = attack_obj_or_fn(env, task)
    else:
        result = attack_obj_or_fn.run(env, task)
    student = env.create_student_model()
    validate_attack_result(result, student)
    student.load_state_dict(result["student_state_dict"])
    p_s, y_s = predict(student, round_ctx["test_x"], cfg, device, batch_size=512)
    am = M.compute_attack_metrics(
        y_s, round_ctx["y_victim"], p_s, round_ctx["p_victim"],
        y_true_perm=round_ctx["test_y_perm"],
    )
    return am, env.defense_session_metrics(), env.query_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", default="soft_label")
    ap.add_argument("--defense", default="smoothing")
    ap.add_argument("--perm-seed", type=int, default=42)
    ap.add_argument("--train-seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--use-submission", action="store_true",
                    help="通过 submission/attack.py 与 submission/defense.py 运行（验证 spec 接口）")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config(os.path.join(root, "config.yaml"))
    if args.quick:
        cfg = apply_quick(cfg)
    device = cfg.get("device", "cpu")

    ev = Evaluator(cfg, device=device)
    print("[match] 准备数据与 Victim ...")
    round_ctx = ev.prepare_round(args.perm_seed, args.train_seed)
    print(f"[match] victim_acc={round_ctx['victim_acc']:.4f}  "
          f"budget={cfg['query']['budget']}  hidden_test={len(round_ctx['test_x'])}")

    # 选择攻击 / 防御来源
    if args.use_submission:
        sys.path.insert(0, root)
        from submission import attack as atkmod, defense as defmod
        attack_obj = atkmod.attack
        is_fn = True
        defense = FunctionDefenseAdapter(defmod.defend, cfg)
        print(f"[match] 使用 submission 接口 (attack={atkmod._ATTACK_NAME}, "
              f"defense={defmod._DEFENSE_NAME})")
    else:
        attack_obj = get_attack(args.attack, cfg)
        is_fn = False
        defense = get_defense(args.defense, cfg)
        print(f"[match] 使用 registry: attack={args.attack} defense={args.defense}")

    # 1) 无防御基线
    print("\n[1] 无防御基线 (No Defense)")
    base_am, _, q0 = _run(attack_obj, get_defense("none", cfg), round_ctx, cfg, device, is_fn)
    la0, pa0 = base_am["label_agreement"], base_am["probability_agreement"]
    print(f"    LA0={la0:.4f}  PA0={pa0:.4f}  score0={base_am['attack_score']:.4f}  "
          f"acc_s={base_am.get('student_accuracy'):.4f}  queries={q0}")

    # 2) 加入防御
    print(f"\n[2] 加入防御: {getattr(defense, 'name', '?')}")
    am, sess, qd = _run(attack_obj, defense, round_ctx, cfg, device, is_fn)
    la_d, pa_d = am["label_agreement"], am["probability_agreement"]
    ler = sess["label_error_rate"]
    sq = sess["service_quality"]
    valid = sess["valid"]

    hard_sup = M.hard_suppression(la0, la_d)
    soft_sup = M.soft_suppression(pa0, pa_d)
    distill_sup = M.distillation_suppression(hard_sup, soft_sup)
    d_score = M.defense_score(distill_sup, sq, ler, cfg["evaluation"]["defense_label_error_limit"])

    print(f"    LA ={la_d:.4f}  PA ={pa_d:.4f}  score ={am['attack_score']:.4f}  "
          f"acc_s={am.get('student_accuracy'):.4f}  queries={qd}")
    print(f"    LabelErrorRate={ler:.4f}  ServiceQuality={sq:.4f}  valid={valid}")
    print(f"    HardSuppression={hard_sup:.4f}  SoftSuppression={soft_sup:.4f}  "
          f"DistillationSuppression={distill_sup:.4f}")
    print(f"    DefenseScore={d_score:.4f}")
    if not valid:
        print("    !! 防御非法：LabelErrorRate 超过 10%，该轮 DefenseScore=0")


if __name__ == "__main__":
    main()
