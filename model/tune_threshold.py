# model/tune_threshold.py

"""
[신설 2026-08-01] target_decision_threshold 스윕·재선정 도구.

eval.py가 남긴 prediction_details_{mark}_{split}.csv 하나만 읽어서
threshold를 바꿔가며 성능을 재계산한다. 모델을 다시 돌리지 않는다.

근거:
  - eval.py의 _decide_with_threshold()는 2-class에서
    Prob(target) >= target_decision_threshold 면 target, 아니면 others 로 판정한다.
    즉 최종 예측은 저장된 Prob_ 컬럼만으로 완전히 재현된다.
  - vild_config.py의 use_others_calibration=False(2026-07-12 확정)이므로
    _apply_others_calibration()은 확률을 변형하지 않고 그대로 통과시킨다.
    따라서 CSV의 Prob_ 컬럼 = 세그먼트 집계 직후의 원본 확률이다.
    (calibration을 다시 켜면 이 전제가 깨지므로 --verify로 반드시 확인할 것.)

의존성은 파이썬 표준 라이브러리뿐이다. torch/pandas/sklearn을 import하지 않는다
(/mnt/c venv에서 torch import는 76초짜리 작업이라 스윕 도구에 얹지 않는다).

사용 예:
  python model/tune_threshold.py --split val
  python model/tune_threshold.py --csv plots/prediction_details_mark4.8_val.csv --min 0.30 --max 0.70
  python model/tune_threshold.py --split val --max_fpr 0.15      # FPR 상한 제약 하에서 고르기
"""

import os
import sys
import csv
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))            # model
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")


def _find_csv(mark_version, split):
    tag = mark_version if split == "test" else "{}_{}".format(mark_version, split)
    cands = [
        os.path.join(PLOTS_DIR, "prediction_details_{}.csv".format(tag)),
        os.path.join(BASE_DIR, "plots", "prediction_details_{}.csv".format(tag)),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "prediction_details CSV를 찾지 못했습니다.\n  찾아본 경로: {}".format("\n            ".join(cands))
    )


def load_rows(csv_path, target_class):
    """CSV를 읽어 (true_label, prob_target, saved_pred) 목록을 만든다."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        prob_col = "Prob_{}".format(target_class)
        if prob_col not in cols:
            prob_cols = [c for c in cols if c.startswith("Prob_")]
            raise KeyError(
                "'{}' 컬럼이 없습니다. CSV에 있는 확률 컬럼: {}\n"
                "  --target 으로 타겟 클래스명을 지정하세요.".format(prob_col, prob_cols)
            )
        rows = []
        for r in reader:
            rows.append((r["True Label"], float(r[prob_col]), r.get("Predicted Label")))
    if not rows:
        raise ValueError("CSV에 데이터 행이 없습니다: {}".format(csv_path))
    return rows


def confusion_at(rows, thr, target_class):
    """threshold thr에서의 혼동행렬. eval.py _decide_with_threshold와 같은 판정식(>=)."""
    tp = fp = fn = tn = 0
    for true_label, prob_t, _ in rows:
        pred_is_target = prob_t >= thr
        true_is_target = (true_label == target_class)
        if true_is_target and pred_is_target:
            tp += 1
        elif true_is_target and not pred_is_target:
            fn += 1
        elif (not true_is_target) and pred_is_target:
            fp += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def metrics_at(rows, thr, target_class):
    tp, fp, fn, tn = confusion_at(rows, thr, target_class)
    n = tp + fp + fn + tn
    acc = (tp + tn) / n if n else 0.0

    t_prec = tp / (tp + fp) if (tp + fp) else 0.0
    t_rec = tp / (tp + fn) if (tp + fn) else 0.0
    t_f1 = 2 * t_prec * t_rec / (t_prec + t_rec) if (t_prec + t_rec) else 0.0

    o_prec = tn / (tn + fn) if (tn + fn) else 0.0
    o_rec = tn / (tn + fp) if (tn + fp) else 0.0
    o_f1 = 2 * o_prec * o_rec / (o_prec + o_rec) if (o_prec + o_rec) else 0.0

    macro_f1 = (t_f1 + o_f1) / 2.0
    # eval.py와 같은 정의: 실제 others인데 타겟으로 오분류된 비율 = 1 - Recall(others)
    others_fpr = fp / (fp + tn) if (fp + tn) else float("nan")

    return {
        "thr": thr, "acc": acc, "macro_f1": macro_f1,
        "t_prec": t_prec, "t_rec": t_rec, "t_f1": t_f1,
        "o_prec": o_prec, "o_rec": o_rec, "o_f1": o_f1,
        "others_fpr": others_fpr,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def roc_auc(rows, target_class):
    """threshold와 무관한 분리력 지표. Mann-Whitney U 방식(동점은 0.5로 처리)."""
    pos = [p for lbl, p, _ in rows if lbl == target_class]
    neg = [p for lbl, p, _ in rows if lbl != target_class]
    if not pos or not neg:
        return float("nan")
    scored = sorted([(p, 1) for p in pos] + [(p, 0) for p in neg], key=lambda x: x[0])
    ranks = {}
    i = 0
    while i < len(scored):
        j = i
        while j + 1 < len(scored) and scored[j + 1][0] == scored[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0          # 1-based 평균 순위
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    rank_sum_pos = sum(ranks[k] for k in range(len(scored)) if scored[k][1] == 1)
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def frange(lo, hi, step):
    """부동소수 누적오차를 피하려고 정수 인덱스로 생성한다."""
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 10) for i in range(n + 1)]


def verify_reproduction(rows, thr, target_class):
    """CSV에 저장된 Predicted Label을 지금 판정식으로 재현할 수 있는지 대조한다.
    100%가 아니면 calibration이 켜져 있거나 저장된 확률이 원본이 아니라는 뜻이다."""
    matched = 0
    checked = 0
    for true_label, prob_t, saved_pred in rows:
        if saved_pred is None:
            continue
        checked += 1
        recomputed = target_class if prob_t >= thr else "others"
        if recomputed == saved_pred:
            matched += 1
    return matched, checked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mark_version", type=str, default="mark4.8")
    ap.add_argument("--split", type=str, default="val", choices=["val", "test"])
    ap.add_argument("--csv", type=str, default=None, help="prediction_details CSV 직접 지정")
    ap.add_argument("--target", type=str, default="dog_bark", help="타겟 클래스명(others가 아닌 쪽)")
    ap.add_argument("--min", type=float, default=0.30, dest="lo")
    ap.add_argument("--max", type=float, default=0.70, dest="hi")
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--max_fpr", type=float, default=None,
                    help="others FPR 상한. 지정하면 이 제약을 만족하는 구간에서만 추천한다")
    ap.add_argument("--verify_thr", type=float, default=None,
                    help="CSV 생성 당시의 config threshold. 지정하면 저장된 예측과 재현 결과를 대조한다")
    ap.add_argument("--out", type=str, default=None, help="스윕 결과 CSV 저장 경로")
    ap.add_argument("--top", type=int, default=12, help="표에 출력할 상위 구간 행 수")
    args = ap.parse_args()

    csv_path = args.csv or _find_csv(args.mark_version, args.split)
    rows = load_rows(csv_path, args.target)
    n_pos = sum(1 for lbl, _, _ in rows if lbl == args.target)
    n_neg = len(rows) - n_pos

    print("=" * 78)
    print("threshold 스윕 — {} / {}".format(args.mark_version, args.split))
    print("  입력: {}".format(csv_path))
    print("  표본: 총 {}개 ({} {}개 / others {}개)".format(len(rows), args.target, n_pos, n_neg))
    print("  ROC AUC: {:.4f}  (threshold와 무관한 분리력)".format(roc_auc(rows, args.target)))
    print("=" * 78)

    if args.verify_thr is not None:
        matched, checked = verify_reproduction(rows, args.verify_thr, args.target)
        pct = (matched / checked * 100) if checked else 0.0
        print("[재현 검증] thr={} 로 재계산한 예측이 CSV의 'Predicted Label'과 {}/{} ({:.1f}%) 일치".format(
            args.verify_thr, matched, checked, pct))
        if matched != checked:
            print("  ⚠️ 100%가 아닙니다. use_others_calibration이 켜져 있거나 저장된 확률이")
            print("     집계 원본이 아닐 수 있습니다. 스윕 결과를 신뢰하기 전에 원인을 확인하세요.")
        print("-" * 78)

    results = [metrics_at(rows, t, args.target) for t in frange(args.lo, args.hi, args.step)]

    # 추천 후보 선정
    pool = results
    if args.max_fpr is not None:
        pool = [r for r in results if r["others_fpr"] <= args.max_fpr]
        if not pool:
            print("[WARN] others FPR <= {} 를 만족하는 threshold가 없습니다. 제약 없이 추천합니다.".format(args.max_fpr))
            pool = results

    # 기존 선정 규칙(2026-07-13~07-15): accuracy 최고 -> 동률이면 macro F1 -> 동률이면 타겟 recall 최고
    best_acc = max(pool, key=lambda r: (round(r["acc"], 6), round(r["macro_f1"], 6), r["t_rec"]))
    best_f1 = max(pool, key=lambda r: (round(r["macro_f1"], 6), round(r["acc"], 6), r["t_rec"]))

    hdr = "  thr    acc   macroF1 | {t}: P     R     F1   | others: P     R     F1   | FPR   | TP  FP  FN  TN".format(
        t=args.target[:7].rjust(7))
    print(hdr)
    print("-" * 78)

    # 최고 accuracy 근처만 표로 출력(전체는 CSV로 저장)
    ranked = sorted(results, key=lambda r: (-round(r["acc"], 6), -round(r["macro_f1"], 6)))
    show_thrs = set(r["thr"] for r in ranked[:args.top])
    show_thrs.add(best_acc["thr"])
    show_thrs.add(best_f1["thr"])
    for r in results:
        if r["thr"] not in show_thrs:
            continue
        mark = ""
        if r["thr"] == best_acc["thr"]:
            mark += " <= acc 최고"
        if r["thr"] == best_f1["thr"] and best_f1["thr"] != best_acc["thr"]:
            mark += " <= macroF1 최고"
        print("  {:.2f}  {:.4f} {:.4f} | {:.3f} {:.3f} {:.3f} | {:.3f} {:.3f} {:.3f} | {:.3f} | {:3d} {:3d} {:3d} {:3d}{}".format(
            r["thr"], r["acc"], r["macro_f1"],
            r["t_prec"], r["t_rec"], r["t_f1"],
            r["o_prec"], r["o_rec"], r["o_f1"],
            r["others_fpr"], r["tp"], r["fp"], r["fn"], r["tn"], mark))

    print("-" * 78)
    print("[추천] accuracy 기준: thr={:.2f}  acc={:.4f}  macroF1={:.4f}  {} R={:.3f}  others FPR={:.3f}".format(
        best_acc["thr"], best_acc["acc"], best_acc["macro_f1"], args.target, best_acc["t_rec"], best_acc["others_fpr"]))
    print("[추천] macroF1 기준: thr={:.2f}  acc={:.4f}  macroF1={:.4f}  {} R={:.3f}  others FPR={:.3f}".format(
        best_f1["thr"], best_f1["acc"], best_f1["macro_f1"], args.target, best_f1["t_rec"], best_f1["others_fpr"]))
    if args.max_fpr is not None:
        print("       (others FPR <= {} 제약 적용)".format(args.max_fpr))

    # 동률 구간을 알려준다 — 어느 폭에서 성능이 평평한지가 운영점 안정성의 근거가 된다
    top_acc = round(best_acc["acc"], 6)
    tie = [r["thr"] for r in pool if round(r["acc"], 6) == top_acc]
    if len(tie) > 1:
        print("       accuracy 동률 구간: {:.2f} ~ {:.2f} ({}개 지점)".format(min(tie), max(tie), len(tie)))

    out_path = args.out or os.path.join(
        PLOTS_DIR, "threshold_sweep_{}_{}.csv".format(args.mark_version, args.split))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "accuracy", "macro_f1",
                    "{}_precision".format(args.target), "{}_recall".format(args.target), "{}_f1".format(args.target),
                    "others_precision", "others_recall", "others_f1",
                    "others_fpr", "TP", "FP", "FN", "TN"])
        for r in results:
            w.writerow(["{:.2f}".format(r["thr"]),
                        "{:.6f}".format(r["acc"]), "{:.6f}".format(r["macro_f1"]),
                        "{:.6f}".format(r["t_prec"]), "{:.6f}".format(r["t_rec"]), "{:.6f}".format(r["t_f1"]),
                        "{:.6f}".format(r["o_prec"]), "{:.6f}".format(r["o_rec"]), "{:.6f}".format(r["o_f1"]),
                        "{:.6f}".format(r["others_fpr"]),
                        r["tp"], r["fp"], r["fn"], r["tn"]])
    print("[저장] 전체 스윕 결과: {}".format(out_path))


if __name__ == "__main__":
    main()
