"""
evaluator.py
============
完整评测流程（spec 第二十四节）。

自动化验证攻击与防御：
  1. 加载 / 划分数据，生成标签置换 π；
  2. 训练 Victim Model（达到准确率门槛）；
  3. 对每个攻击运行无防御基线 -> LA0 / PA0（攻击资格 + 抑制基准）；
  4. 对每个 (Attack × Defense) 组合重新执行蒸馏 -> LA / PA / 服务质量；
  5. 计算 AttackScore / DistillationSuppression / DefenseScore；
  6. 汇总 FinalAttackScore / FinalDefenseScore，输出 JSON + CSV 报告。
"""

from __future__ import annotations

import csv
import json
import os
import time

import numpy as np
import torch

# 导入即注册全部攻击 / 防御
import attacks as _attacks_pkg  # noqa: F401
import defenses as _defenses_pkg  # noqa: F401
from attacks import get_attack
from defenses import get_defense

from .dataset import build_splits
from .label_mapping import apply_mapping, generate_permutation
from .train_victim import save_victim, train_victim
from .training_utils import predict
from .environment import DistillationEnv
from .validator import BudgetExceededError, IllegalAttackOutputError, validate_attack_result
from . import metrics as M


class Evaluator:
    def __init__(self, cfg: dict, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.results_dir = cfg.get("results_dir", "./results")
        os.makedirs(self.results_dir, exist_ok=True)
        self.ckpt_dir = cfg["victim"]["checkpoint_dir"]
        os.makedirs(self.ckpt_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 数据与 Victim 准备
    # ------------------------------------------------------------------ #
    def prepare_round(self, perm_seed: int, train_seed: int):
        """准备一轮评测所需的数据、标签置换与 Victim Model。"""
        splits = build_splits(self.cfg, seed=perm_seed)
        mapping = generate_permutation(perm_seed, num_classes=10)

        victim_y_perm = apply_mapping(splits.victim_train_y, mapping)
        victim, acc = self._get_or_train_victim(
            perm_seed, train_seed, splits.victim_train_x, victim_y_perm, mapping)

        # 隐藏测试集（可子采样加速）
        tx, ty = splits.hidden_test_x, apply_mapping(splits.hidden_test_y, mapping)
        subset = self.cfg["evaluation"].get("eval_subset")
        if subset and len(tx) > subset:
            rng = np.random.default_rng(perm_seed + 1)
            idx = rng.choice(len(tx), size=subset, replace=False)
            tx, ty = tx[idx], ty[idx]

        # Victim 在隐藏测试集上的预测（参考行为，仅裁判持有）
        p_v, y_v = predict(victim, tx, self.cfg, self.device, batch_size=512)
        return {
            "splits": splits, "mapping": mapping, "victim": victim,
            "victim_acc": acc, "test_x": tx, "test_y_perm": ty,
            "p_victim": p_v, "y_victim": y_v,
        }

    def _get_or_train_victim(self, perm_seed, train_seed, vx, vy_perm, mapping):
        path = os.path.join(self.ckpt_dir, f"victim_p{perm_seed}_t{train_seed}.pt")
        victim_train_size = int(self.cfg["data"]["splits"]["victim_train"])
        if os.path.exists(path):
            try:
                ckpt = torch.load(path, map_location=self.device, weights_only=False)
                from .train_victim import victim_checkpoint_matches
                if victim_checkpoint_matches(ckpt, mapping, self.cfg, victim_train_size):
                    from .victim_model import build_victim
                    victim = build_victim(ckpt["arch"], num_classes=10).to(self.device)
                    victim.load_state_dict(ckpt["state_dict"])
                    victim.eval()
                    print(f"[eval] 复用 Victim checkpoint (acc={ckpt['acc']:.4f})")
                    return victim, ckpt["acc"]
                print("[eval] checkpoint 配置不匹配（规模/轮数/标签置换），重新训练")
            except Exception as e:  # noqa: BLE001
                print(f"[eval] 加载 checkpoint 失败: {e}，重新训练")

        victim, acc = train_victim(self.cfg, vx, vy_perm, self.device, seed=train_seed)
        save_victim(victim, acc, mapping, self.cfg, path)
        return victim, acc

    # ------------------------------------------------------------------ #
    # 单次攻击运行
    # ------------------------------------------------------------------ #
    def _build_task(self, splits):
        return {
            "query_images": splits.query_pool_x,
            "query_budget": self.cfg["query"]["budget"],
            "num_classes": 10,
            "image_shape": (1, 28, 28),
        }

    def _run_attack(self, attack_name, defense, round_ctx):
        """运行一次攻击（含指定防御），返回 (student_metrics, env_session_metrics, status)。"""
        env = DistillationEnv(
            round_ctx["victim"], defense, round_ctx["splits"].query_pool_x,
            self.cfg["query"]["budget"], self.cfg, self.device, num_classes=10,
        )
        task = self._build_task(round_ctx["splits"])
        attack = get_attack(attack_name, self.cfg)
        try:
            result = attack.run(env, task)
            student = env.create_student_model()
            validate_attack_result(result, student)
            student.load_state_dict(result["student_state_dict"])
            p_s, y_s = predict(student, round_ctx["test_x"], self.cfg, self.device, batch_size=512)
            am = M.compute_attack_metrics(
                y_s, round_ctx["y_victim"], p_s, round_ctx["p_victim"],
                y_true_perm=round_ctx["test_y_perm"],
            )
            sess = env.defense_session_metrics()
            return am, sess, "ok"
        except BudgetExceededError as e:
            return ({"label_agreement": 0.1, "probability_agreement": 0.0,
                     "attack_score": 0.08, "student_accuracy": 0.1},
                    env.defense_session_metrics(), f"budget_exceeded: {e}")
        except (IllegalAttackOutputError, Exception) as e:  # noqa: BLE001
            return ({"label_agreement": 0.1, "probability_agreement": 0.0,
                     "attack_score": 0.08, "student_accuracy": 0.1},
                    env.defense_session_metrics() if "env" in dir() else {},
                    f"failed: {e}")

    # ------------------------------------------------------------------ #
    # 主评测
    # ------------------------------------------------------------------ #
    def run_full_evaluation(self, attack_names=None, defense_names=None,
                            num_permutations=None, num_seeds=None) -> dict:
        cfg = self.cfg
        attack_names = attack_names or cfg["methods"]["attacks"]
        defense_names = defense_names or cfg["methods"]["defenses"]
        num_perm = num_permutations or cfg["evaluation"]["num_permutations"]
        num_seeds = num_seeds or cfg["evaluation"]["num_seeds"]
        base_seed = int(cfg.get("seed", 42))
        error_limit = cfg["evaluation"]["defense_label_error_limit"]

        perm_results = []
        # 累加器用于最终平均
        attack_score_acc = {a: [] for a in attack_names}      # 每个 attack 的 AttackScore（含各 defense）
        defense_score_acc = {d: [] for d in defense_names}     # 每个 defense 的 DefenseScore（含各 attack）
        baseline_acc = {a: {"LA": [], "PA": [], "score": []} for a in attack_names}

        for pi in range(num_perm):
            for si in range(num_seeds):
                perm_seed = base_seed + pi * 101
                train_seed = base_seed + si * 17
                tag = f"perm={perm_seed},seed={train_seed}"
                print(f"\n{'='*60}\n[eval] 轮次 {pi*num_seeds+si+1}/{num_perm*num_seeds}  {tag}\n{'='*60}")
                t0 = time.time()
                round_ctx = self.prepare_round(perm_seed, train_seed)
                print(f"[eval] victim_acc={round_ctx['victim_acc']:.4f}  "
                      f"hidden_test={len(round_ctx['test_x'])}")

                # 1) 无防御基线（每个攻击）
                baselines = {}
                for a in attack_names:
                    print(f"\n--- 基线攻击: {a} (No Defense) ---")
                    defense_none = get_defense("none", cfg)
                    am, sess, status = self._run_attack(a, defense_none, round_ctx)
                    baselines[a] = {
                        "label_agreement": am["label_agreement"],
                        "probability_agreement": am["probability_agreement"],
                        "attack_score": am["attack_score"],
                        "student_accuracy": am.get("student_accuracy"),
                        "query_count": sess.get("num_queries", 0),
                        "status": status,
                    }
                    baseline_acc[a]["LA"].append(am["label_agreement"])
                    baseline_acc[a]["PA"].append(am["probability_agreement"])
                    baseline_acc[a]["score"].append(am["attack_score"])
                    print(f"    LA0={am['label_agreement']:.4f} PA0={am['probability_agreement']:.4f} "
                          f"score0={am['attack_score']:.4f} [{status}]")

                # 2) 攻防全对阵矩阵
                matrix = {a: {} for a in attack_names}
                for a in attack_names:
                    la0 = baselines[a]["label_agreement"]
                    pa0 = baselines[a]["probability_agreement"]
                    for d in defense_names:
                        if d == "none":
                            am = {"label_agreement": la0, "probability_agreement": pa0,
                                  "attack_score": baselines[a]["attack_score"],
                                  "student_accuracy": baselines[a].get("student_accuracy")}
                            sess = {"label_error_rate": 0.0, "service_quality": 1.0,
                                    "valid": True, "num_queries": baselines[a]["query_count"]}
                            status = "baseline_reuse"
                        else:
                            print(f"\n--- 对阵: {a}  ×  {d} ---")
                            defense = get_defense(d, cfg)
                            am, sess, status = self._run_attack(a, defense, round_ctx)

                        la_d = am["label_agreement"]
                        pa_d = am["probability_agreement"]
                        ler = sess.get("label_error_rate", 0.0)
                        sq = sess.get("service_quality", 1.0)
                        valid = sess.get("valid", True)

                        hard_sup = M.hard_suppression(la0, la_d)
                        soft_sup = M.soft_suppression(pa0, pa_d)
                        distill_sup = M.distillation_suppression(hard_sup, soft_sup)
                        d_score = M.defense_score(distill_sup, sq, ler, error_limit)

                        cell = {
                            "label_agreement": la_d,
                            "probability_agreement": pa_d,
                            "attack_score": am["attack_score"],
                            "student_accuracy": am.get("student_accuracy"),
                            "label_error_rate": ler,
                            "service_quality": sq,
                            "defense_valid": valid,
                            "hard_suppression": hard_sup,
                            "soft_suppression": soft_sup,
                            "distillation_suppression": distill_sup,
                            "defense_score": d_score,
                            "query_count": sess.get("num_queries", 0),
                            "status": status,
                        }
                        matrix[a][d] = cell
                        print(f"    LA={la_d:.4f} PA={pa_d:.4f} score={am['attack_score']:.4f} | "
                              f"LER={ler:.4f} SQ={sq:.4f} DSup={distill_sup:.4f} "
                              f"DScore={d_score:.4f} [{status}]")

                        attack_score_acc[a].append(am["attack_score"])
                        defense_score_acc[d].append(d_score)

                perm_results.append({
                    "perm_seed": perm_seed, "train_seed": train_seed,
                    "victim_acc": round_ctx["victim_acc"],
                    "baselines": baselines, "matrix": matrix,
                })
                print(f"\n[eval] 轮次耗时 {time.time()-t0:.1f}s")

        # 3) 汇总最终得分
        qual = cfg["evaluation"]["attack_qualification"]
        final_attacks = {}
        for a in attack_names:
            mean_la0 = float(np.mean(baseline_acc[a]["LA"])) if baseline_acc[a]["LA"] else 0.0
            mean_sc0 = float(np.mean(baseline_acc[a]["score"])) if baseline_acc[a]["score"] else 0.0
            qualified = (mean_la0 >= qual["min_label_agreement"]
                         and mean_sc0 >= qual["min_attack_score"])
            final_attacks[a] = {
                "final_attack_score": float(np.mean(attack_score_acc[a])) if attack_score_acc[a] else 0.0,
                "baseline_label_agreement": mean_la0,
                "baseline_attack_score": mean_sc0,
                "qualified": qualified,
            }
        final_defenses = {
            d: {"final_defense_score": float(np.mean(defense_score_acc[d])) if defense_score_acc[d] else 0.0}
            for d in defense_names
        }

        report = {
            "config_summary": {
                "query_budget": cfg["query"]["budget"],
                "attacks": attack_names, "defenses": defense_names,
                "num_permutations": num_perm, "num_seeds": num_seeds,
                "label_error_limit": error_limit,
            },
            "permutations": perm_results,
            "final": {"attacks": final_attacks, "defenses": final_defenses},
        }
        self._save_report(report)
        self._print_summary(report)
        return report

    # ------------------------------------------------------------------ #
    # 输出
    # ------------------------------------------------------------------ #
    def _save_report(self, report: dict):
        ts = int(time.time())  # 仅用于文件名，评测本身不依赖时间
        json_path = os.path.join(self.results_dir, f"report_{ts}.json")
        csv_path = os.path.join(self.results_dir, f"report_{ts}.csv")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        self._write_csv(report, csv_path)
        print(f"\n[eval] 报告已保存: {json_path}\n          CSV: {csv_path}")

    def _write_csv(self, report: dict, path: str):
        rows = []
        for pr in report["permutations"]:
            for a, dmap in pr["matrix"].items():
                for d, c in dmap.items():
                    rows.append({
                        "perm_seed": pr["perm_seed"], "train_seed": pr["train_seed"],
                        "victim_acc": pr["victim_acc"],
                        "attack": a, "defense": d,
                        **c,
                    })
        if not rows:
            return
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    def _print_summary(self, report: dict):
        print("\n" + "=" * 60)
        print("最终排名汇总")
        print("=" * 60)
        fa = report["final"]["attacks"]
        print("\n[攻击方] FinalAttackScore（越高越好）")
        for a, v in sorted(fa.items(), key=lambda x: -x[1]["final_attack_score"]):
            tag = "[OK] 合格" if v["qualified"] else "[X] 未达资格线"
            print(f"  {a:<14} score={v['final_attack_score']:.4f}  "
                  f"(baseline LA={v['baseline_label_agreement']:.4f}, "
                  f"score0={v['baseline_attack_score']:.4f}) {tag}")
        fd = report["final"]["defenses"]
        print("\n[防御方] FinalDefenseScore（越高越好；>0.10 LER 则该格为 0）")
        for d, v in sorted(fd.items(), key=lambda x: -x[1]["final_defense_score"]):
            print(f"  {d:<14} score={v['final_defense_score']:.4f}")
        print("=" * 60)
