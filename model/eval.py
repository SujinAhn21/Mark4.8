# model/eval.py  

"""
[Deprecated: 평가용 샘플을 per_class_max=30으로 샘플링하던 로직]
# sampled_files = []; per_class_max = 30; class_counter = defaultdict(int); ...
"""
import os, sys, csv, glob, torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np, pandas as pd, argparse
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_fscore_support, accuracy_score,
                             roc_auc_score, roc_curve, auc)
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))         # model
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
UTILS_DIR = os.path.join(PROJECT_ROOT, 'utils')
VILD_DIR = os.path.join(PROJECT_ROOT, 'vild')
for p in (PROJECT_ROOT, UTILS_DIR, VILD_DIR):
    if p not in sys.path: sys.path.append(p)

from vild_config import AudioViLDConfig
from vild_model import SimpleAudioEncoder, ViLDTextHead, LearnableBackgroundEmbedding
from vild_head import DualBranchStudentHead
from vild_parser_student import AudioParser
from seed_utils import set_seed
SHARED_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "shared_vild"))
if SHARED_DIR not in sys.path:
    sys.path.append(SHARED_DIR)
from checkpoint_utils import load_checkpoint, resolve_state_dict

def _find_dataset_index(mark_version):
    for p in [os.path.join(PROJECT_ROOT, f"dataset_index_{mark_version}.csv"),
              os.path.join(BASE_DIR,     f"dataset_index_{mark_version}.csv")]:
        if os.path.exists(p): return p
    raise FileNotFoundError("dataset_index CSV not found.")

def _find_student_weights(mark_version):
    enc_primary = [os.path.join(BASE_DIR, f"distilled_student_encoder_{mark_version}.pth"),
                   os.path.join(BASE_DIR, f"best_student_encoder_{mark_version}.pth")]
    head_primary = [os.path.join(BASE_DIR, f"distilled_student_head_{mark_version}.pth"),
                    os.path.join(BASE_DIR, f"best_student_head_{mark_version}.pth")]
    CWD = os.getcwd()
    extra_roots = [PROJECT_ROOT, CWD]
    enc_extra, head_extra = [], []
    for r in extra_roots:
        enc_extra += [os.path.join(r, f"distilled_student_encoder_{mark_version}.pth"),
                      os.path.join(r, f"best_student_encoder_{mark_version}.pth")]
        head_extra += [os.path.join(r, f"distilled_student_head_{mark_version}.pth"),
                       os.path.join(r, f"best_student_head_{mark_version}.pth")]
    glob_roots = [PROJECT_ROOT, "/content"]
    enc_glob, head_glob = [], []
    for r in glob_roots:
        enc_glob += glob.glob(os.path.join(r, f"**/distilled_student_encoder_{mark_version}.pth"), recursive=True)
        enc_glob += glob.glob(os.path.join(r, f"**/best_student_encoder_{mark_version}.pth"), recursive=True)
        head_glob += glob.glob(os.path.join(r, f"**/distilled_student_head_{mark_version}.pth"), recursive=True)
        head_glob += glob.glob(os.path.join(r, f"**/best_student_head_{mark_version}.pth"), recursive=True)
    enc_candidates = enc_primary + enc_extra + enc_glob
    head_candidates= head_primary+ head_extra+ head_glob
    enc = next((p for p in enc_candidates if os.path.exists(p)), None)
    hed = next((p for p in head_candidates if os.path.exists(p)), None)
    return enc, hed, enc_candidates, head_candidates


def _compute_segment_saliency(seg_tensor):
    if seg_tensor.ndim == 4:
        seg_tensor = seg_tensor.squeeze(0)
    base = seg_tensor[0] if seg_tensor.ndim == 3 else seg_tensor
    energy = float(base.abs().mean().item())
    if base.shape[-1] > 1:
        flux = float((base[:, 1:] - base[:, :-1]).abs().mean().item())
    else:
        flux = 0.0
    return energy + 0.5 * flux


def _aggregate_segment_probs(segment_probs, saliency_scores, config):
    probs = np.asarray(segment_probs, dtype=np.float32)
    saliency = np.asarray(saliency_scores, dtype=np.float32)
    if probs.ndim != 2 or len(probs) == 0:
        raise ValueError("segment_probs must be a non-empty [K, C] array")

    if config.segment_aggregation_mode == "mean":
        weights = np.ones(len(probs), dtype=np.float32)
    else:
        confidence = probs.max(axis=1)
        conf_weights = np.power(np.clip(confidence, 1e-6, 1.0), config.segment_confidence_power)
        saliency_norm = saliency / max(float(saliency.max()), 1e-6)
        saliency_weights = np.power(np.clip(saliency_norm, 1e-6, 1.0), config.segment_saliency_power)
        weights = conf_weights * saliency_weights

    weights = weights / max(float(weights.sum()), 1e-6)
    aggregated = (probs * weights[:, None]).sum(axis=0)
    return aggregated, weights


def _decide_with_threshold(prob_vec, class_names, config):
    """
    [추가 2026-07-13] 최종 예측 결정. 2-class(타겟 vs others)에서는 argmax(=암묵적 0.5) 대신
    config.target_decision_threshold를 쓴다: Prob(타겟) >= threshold면 타겟, 아니면 others.
    근거: 가설 1·2·3·4·6 재학습 후 모델이 초보수적(dog precision 1.0/FPR 0.0)이 되어 놓친
    dog 19건 중 11건이 0.4~0.5 구간에 몰림 -> threshold 0.40으로 recall 0.62->0.84 회복
    (상세는 vild_config.py 주석). threshold가 None이거나 2-class가 아니면 기존 argmax.
    """
    thr = getattr(config, "target_decision_threshold", None)
    if thr is not None and len(class_names) == 2 and "others" in class_names:
        others_idx = class_names.index("others")
        target_idx = 1 - others_idx
        return target_idx if float(prob_vec[target_idx]) >= thr else others_idx
    return int(np.argmax(prob_vec))


def _apply_others_calibration(prob_vec, class_names, config):
    calibrated = prob_vec.copy()
    others_idx = class_names.index("others")
    top_idx = int(np.argmax(calibrated))
    sorted_idx = np.argsort(calibrated)[::-1]
    top_conf = float(calibrated[top_idx])
    second_conf = float(calibrated[sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0
    margin = top_conf - second_conf

    entropy = float(-(calibrated * np.log(np.clip(calibrated, 1e-8, 1.0))).sum() / np.log(len(class_names)))
    # [확정 2026-07-12] use_others_calibration=False면 override 자체를 건너뛴다(raw 예측 유지).
    # mark4.8(2-class)에서 이 override가 dog_bark recall만 깎아(62%->42%) 비활성이 맞다고 실측 확정.
    force_others = (
        getattr(config, "use_others_calibration", True)
        and top_idx != others_idx
        and (
            top_conf < config.others_confidence_threshold
            or margin < config.others_margin_threshold
            or entropy > config.others_entropy_threshold
        )
    )
    if force_others:
        boosted = calibrated.copy()
        boosted[others_idx] = max(boosted[others_idx], top_conf + 1e-3)
        boosted = boosted / boosted.sum()
        return boosted, others_idx, {
            "forced_to_others": True,
            "raw_top_conf": top_conf,
            "raw_margin": margin,
            "entropy": entropy,
        }

    return calibrated, top_idx, {
        "forced_to_others": False,
        "raw_top_conf": top_conf,
        "raw_margin": margin,
        "entropy": entropy,
    }


def _save_visual_explanation(path, segments, segment_probs, segment_weights, class_names, final_prob, final_pred, config, plot_dir):
    if not config.save_visual_explanations:
        return

    explanation_dir = os.path.join(plot_dir, f"explanations_{config.mark_version}")
    os.makedirs(explanation_dir, exist_ok=True)

    order = np.argsort(segment_weights)[::-1][:config.explain_topk_segments]
    fig, axes = plt.subplots(len(order), 1, figsize=(10, 3 * len(order)))
    if len(order) == 1:
        axes = [axes]

    for ax, idx in zip(axes, order):
        seg = segments[idx]
        if seg.ndim == 4:
            seg = seg.squeeze(0)
        base = seg[0].cpu().numpy() if seg.ndim == 3 else seg.cpu().numpy()
        sns.heatmap(base, ax=ax, cmap="magma", cbar=True)
        pred_idx = int(np.argmax(segment_probs[idx]))
        ax.set_title(
            f"seg#{idx} weight={segment_weights[idx]:.3f} "
            f"pred={class_names[pred_idx]} conf={segment_probs[idx][pred_idx]:.3f}"
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Mel")

    fig.suptitle(
        f"{os.path.basename(path)} | final={class_names[final_pred]} "
        f"| probs={np.array2string(final_prob, precision=3, suppress_small=True)}",
        fontsize=10,
    )
    fig.tight_layout()
    out_path = os.path.join(explanation_dir, f"{os.path.splitext(os.path.basename(path))[0]}.png")
    fig.savefig(out_path)
    plt.close(fig)

def evaluate(audio_label_list, seed_value=42, mark_version="mark4.1"):
    set_seed(seed_value)
    config = AudioViLDConfig(mark_version=mark_version)
    parser = AudioParser(config)
    device = config.device

    cls = config.get_classes_for_text_prompts(); ncls = len(cls)
    idx2 = {i: l for i,l in enumerate(cls)}
    lab2 = {l: i for i,l in enumerate(cls)}

    enc = SimpleAudioEncoder(config).to(device)
    branch_head = DualBranchStudentHead(config.embedding_dim).to(device)
    head= ViLDTextHead(config).to(device)
    text_emb = config.get_class_text_embeddings().to(device)

    enc_path, head_path, ec, hc = _find_student_weights(mark_version)
    if not enc_path or not head_path:
        print(f"[ERROR] 모델 파일을 찾지 못했습니다.\n  - enc: {ec}\n  - head: {hc}")
        return
    enc_ckpt = load_checkpoint(enc_path, map_location=device)
    head_ckpt = load_checkpoint(head_path, map_location=device)
    enc.load_state_dict(resolve_state_dict(enc_ckpt, "model_state_dict", "encoder_state_dict", "model"))
    branch_head.load_state_dict(resolve_state_dict(head_ckpt, "branch_state_dict", "head_state_dict", "head"), strict=False)
    enc.eval(); branch_head.eval(); head.eval()

    background_embedding = None
    if getattr(config, "use_background_embedding", False):
        # [수정] background_embedding은 head_ckpt(EarlyStopping이 encoder/head와 같은 시점에
        # 저장한 파일)에서 읽는다. 별도 체크포인트(학습 종료 시점=마지막 epoch)에서 읽으면
        # early stopping으로 encoder/head가 더 이른 epoch에 고정된 경우 서로 다른 epoch의
        # 파라미터가 짝지어지는 버그가 생기기 때문.
        bg_state = head_ckpt.get("background_embedding_state_dict")
        if bg_state is not None:
            background_embedding = LearnableBackgroundEmbedding(config.embedding_dim).to(device)
            background_embedding.load_state_dict(bg_state)
            background_embedding.eval()
        else:
            print(
                "[WARN] background_embedding_state_dict를 찾지 못했습니다 "
                f"(checkpoint: {head_path}). 옛 체크포인트로 간주하고 background embedding 보정을 건너뜁니다."
            )

    def _apply_background_override(logits, features):
        others_idx = cls.index("others")
        region_norm = F.normalize(features, dim=1)
        bg_norm = F.normalize(background_embedding(), dim=0)
        bg_logit = (region_norm @ bg_norm) / head.temperature
        logits = logits.clone()
        logits[:, others_idx] = torch.maximum(logits[:, others_idx], bg_logit)
        return logits

    y_true, y_pred, y_prob, paths = [], [], [], []
    raw_y_pred, raw_y_prob = [], []
    calibration_logs = []
    for path, tlabel in audio_label_list:
        if tlabel not in lab2: continue
        tidx = lab2[tlabel]
        segs = parser.load_and_segment(path)
        if not segs:
            print(f"[INFO] Skip (no valid segments): {os.path.basename(path)}"); continue
        segment_probs = []
        saliency_scores = []
        with torch.no_grad():
            for seg in segs:
                if seg is None or seg.ndim not in (3,4): continue
                if seg.ndim == 3: seg = seg.unsqueeze(0)
                seg = seg.to(device)
                feat = enc(seg)
                supervised_features, distill_features = branch_head(feat)
                sup_logits = head(supervised_features, text_emb)
                distill_logits = head(distill_features, text_emb)
                if background_embedding is not None:
                    sup_logits = _apply_background_override(sup_logits, supervised_features)
                    distill_logits = _apply_background_override(distill_logits, distill_features)
                w = getattr(config, "distill_branch_eval_weight", 0.5)
                prob = (
                    (1 - w) * torch.softmax(sup_logits, dim=-1)
                    + w * torch.softmax(distill_logits, dim=-1)
                ).squeeze(0)
                segment_probs.append(prob.cpu().numpy())
                saliency_scores.append(_compute_segment_saliency(seg.cpu()))
        if not segment_probs:
            continue

        aggregated, seg_weights = _aggregate_segment_probs(segment_probs, saliency_scores, config)
        # [수정 2026-07-13] raw/최종 예측 모두 threshold 결정 적용(기존 argmax).
        # others-calibration이 개입해 others로 강등한 경우(forced)만 그 결과를 유지한다.
        raw_pred = _decide_with_threshold(aggregated, cls, config)
        calibrated_prob, pred, cal_meta = _apply_others_calibration(aggregated, cls, config)
        if not cal_meta["forced_to_others"]:
            pred = _decide_with_threshold(calibrated_prob, cls, config)

        y_true.append(tidx)
        y_pred.append(pred)
        y_prob.append(calibrated_prob)
        raw_y_pred.append(raw_pred)
        raw_y_prob.append(aggregated)
        calibration_logs.append(cal_meta)
        paths.append(path)

        _save_visual_explanation(
            path,
            segs,
            segment_probs,
            seg_weights,
            cls,
            calibrated_prob,
            pred,
            config,
            os.path.join(PROJECT_ROOT, "plots"),
        )

    if not y_true:
        print("[WARN] 평가 가능한 예측 없음."); return

    y_true = np.array(y_true); y_pred = np.array(y_pred); y_prob = np.array(y_prob)
    raw_y_pred = np.array(raw_y_pred); raw_y_prob = np.array(raw_y_prob)
    plot_dir = os.path.join(PROJECT_ROOT, "plots"); os.makedirs(plot_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(ncls)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cls)
    disp.plot(cmap=plt.cm.Blues); plt.title(f"Confusion Matrix ({mark_version})")
    plt.tight_layout(); plt.savefig(os.path.join(plot_dir, f"confusion_matrix_{mark_version}.png")); plt.close()
    print("[INFO] Confusion matrix 저장 완료.")

    acc = accuracy_score(y_true, y_pred)
    pre, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None,
                                                      labels=list(range(ncls)), zero_division=0)
    if ncls==2:
        rocA = roc_auc_score(y_true, y_prob[:,1])
    else:
        rocA = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')

    # [추가] others FPR: 실제 라벨이 "others"인데 타겟 클래스로 오분류된 비율 (= 1 - Recall(others)).
    others_idx = cls.index("others")
    others_row = cm[others_idx]
    others_row_sum = int(others_row.sum())
    others_fpr = float((others_row_sum - others_row[others_idx]) / others_row_sum) if others_row_sum > 0 else float('nan')
    print(f"[others FPR] 실제 others인데 타겟 클래스로 오분류된 비율: {others_fpr:.4f}")

    print("\n" + "="*30)
    print(f"      성능 평가 결과 ({mark_version})")
    print("="*30)
    print(f"  - Accuracy: {acc:.4f}")
    if isinstance(rocA, float): print(f"  - ROC AUC: {rocA:.4f}")
    print("\n클래스별 성능:")
    for i in range(ncls):
        print(f"  - {cls[i]} | P:{pre[i]:.4f} R:{rec[i]:.4f} F1:{f1[i]:.4f}")
    print("="*30 + "\n")

    data = {'Precision': list(pre)+[None], 'Recall': list(rec)+[None], 'F1-Score': list(f1)+[None]}
    df = pd.DataFrame(data, index=cls+['Overall'])
    df.loc['Overall','Accuracy']=acc; df.loc['Overall','ROC AUC']=rocA if isinstance(rocA,float) else None
    plt.figure(figsize=(8,4))
    sns.heatmap(df, annot=True, fmt=".4f", cmap="viridis", cbar=False, linewidths=.5)
    plt.title(f'Performance Metrics ({mark_version})'); plt.xticks(fontsize=12); plt.yticks(fontsize=12, rotation=0)
    plt.tight_layout(); plt.savefig(os.path.join(plot_dir, f'performance_metrics_table_{mark_version}.png')); plt.close()

    plt.figure(figsize=(7,6))
    if ncls==2:
        fpr,tpr,_ = roc_curve(y_true, y_prob[:,1]); plt.plot(fpr,tpr, label=f'AUC={rocA:.4f}')
    else:
        for i in range(ncls):
            fpr,tpr,_ = roc_curve(y_true==i, y_prob[:,i]); A = auc(fpr,tpr)
            plt.plot(fpr,tpr, label=f'{cls[i]} AUC={A:.4f}')
    plt.plot([0,1],[0,1],'k--'); plt.xlim([0,1]); plt.ylim([0,1.05])
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC ({mark_version})'); plt.legend(loc="lower right")
    plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(plot_dir, f'roc_curve_{mark_version}.png')); plt.close()

    csv_path = os.path.join(plot_dir, f'performance_summary_{mark_version}.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        f.write(f"# Performance Summary for {mark_version}\n\n")
        pd.DataFrame({'Metric':['Accuracy','ROC AUC' if ncls==2 else 'ROC AUC (Macro)', 'Others FPR'],
                      'Score':[acc, rocA if isinstance(rocA,float) else 'N/A', others_fpr]}).to_csv(f, index=False)
        f.write("\n# Class-wise Metrics\n\n")
        pd.DataFrame({'Class':cls,'Precision':pre,'Recall':rec,'F1-Score':f1}).to_csv(f, index=False)
    print(f"[INFO] 성능 요약 CSV 저장: {csv_path}")

    pred_results = os.path.join(plot_dir, f'prediction_details_{mark_version}.csv')
    with open(pred_results, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([
            'Filename', 'True Label', 'Raw Predicted Label', 'Predicted Label',
            'Forced To Others', 'Raw Top Confidence', 'Raw Margin', 'Entropy'
        ] + [f'Prob_{n}' for n in cls])
        for i in range(len(paths)):
            meta = calibration_logs[i]
            w.writerow([
                os.path.basename(paths[i]),
                idx2[y_true[i]],
                idx2[raw_y_pred[i]],
                idx2[y_pred[i]],
                meta['forced_to_others'],
                meta['raw_top_conf'],
                meta['raw_margin'],
                meta['entropy'],
            ] + list(y_prob[i]))
    print(f"[INFO] 상세 예측 결과 CSV 저장: {pred_results}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mark_version', type=str, required=True)
    args = parser.parse_args()

    config = AudioViLDConfig(mark_version=args.mark_version)
    csv_path = _find_dataset_index(args.mark_version)
    pre_parser = AudioParser(config)

    # test 전량 사용, 샘플링 제거
    data = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            p, l = row['path'], row['label']
            if l in config.classes and ("/data/test/" in p.replace("\\","/")):
                data.append((p, l))

    print(f"[INFO] test 전량 후보: {len(data)}개. 유효성 검사 후 평가 시작.")
    valid = []
    for path, label in data:
        segs = pre_parser.load_and_segment(path)
        if segs: valid.append((path, label))
        else: print(f"[WARN] 무효 파일 제외: {os.path.basename(path)}")

    print(f"[INFO] 유효 test 샘플: {len(valid)}개")
    if not valid:
        print("[ERROR] 평가할 유효 test 샘플이 없습니다.")
    else:
        evaluate(valid, seed_value=42, mark_version=args.mark_version)
    
