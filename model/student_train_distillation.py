# model/student_train_distillation.py

import os
import sys
import csv
import pickle
import argparse
import functools

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

print = functools.partial(print, flush=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
VILD_DIR = os.path.join(PROJECT_ROOT, "vild")
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
for p in (PROJECT_ROOT, VILD_DIR, UTILS_DIR):
    if p not in sys.path:
        sys.path.append(p)

from vild_config import AudioViLDConfig
from vild_model import SimpleAudioEncoder, ViLDTextHead, LearnableBackgroundEmbedding
from vild_head import DualBranchStudentHead
from vild_parser_student import AudioParser
from seed_utils import set_seed
SHARED_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "shared_vild"))
if SHARED_DIR not in sys.path:
    sys.path.append(SHARED_DIR)
from checkpoint_utils import save_checkpoint


class EarlyStopping:
    def __init__(self, patience=10, verbose=True, delta=0, path_encoder="encoder.pth", path_head="head.pth", mark_version="unknown"):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")
        self.path_encoder = path_encoder
        self.path_head = path_head
        self.mark_version = mark_version

    def __call__(self, val_loss, encoder, head, background_embedding=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save(val_loss, encoder, head, background_embedding)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save(val_loss, encoder, head, background_embedding)
            self.counter = 0

    def _save(self, val_loss, encoder, head, background_embedding=None):
        if self.verbose:
            print(f"[EarlyStopping] Val loss {self.val_loss_min:.6f} -> {val_loss:.6f}. Saving...")
        save_checkpoint(self.path_encoder, "student_encoder", self.mark_version, model_state=encoder.state_dict())
        # [수정] background_embedding을 head와 같은 파일에, 같은 시점(best val epoch)에 저장.
        # encoder/head는 best epoch에 저장되는데 background_embedding만 학습 종료 시점(마지막 epoch)에
        # 별도 저장하면 서로 다른 epoch의 파라미터가 eval 시 짝지어지는 버그가 생기기 때문.
        background_state = background_embedding.state_dict() if background_embedding is not None else None
        save_checkpoint(
            self.path_head, "student_branch", self.mark_version,
            branch_state=head.state_dict(), background_embedding_state=background_state,
        )
        self.val_loss_min = val_loss


class DistillationLoss(nn.Module):
    def __init__(self, T, alpha, feature_kd_weight=0.0, ignore_index=-1):
        super().__init__()
        self.T = T
        self.alpha = alpha
        self.feature_kd_weight = feature_kd_weight
        self.hard = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.soft = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, soft_labels, hard_labels, student_features=None, teacher_features=None):
        valid = hard_labels != -1
        if not valid.any():
            return {
                "total": torch.tensor(0.0, device=student_logits.device, requires_grad=True),
                "hard": torch.tensor(0.0, device=student_logits.device),
                "soft": torch.tensor(0.0, device=student_logits.device),
                "feature": torch.tensor(0.0, device=student_logits.device),
            }

        loss_hard = self.hard(student_logits[valid], hard_labels[valid])
        s = F.log_softmax(student_logits[valid] / self.T, dim=1)
        t = F.softmax(soft_labels[valid] / self.T, dim=1)
        loss_soft = self.soft(s, t) * (self.T ** 2)

        loss_feature = torch.tensor(0.0, device=student_logits.device)
        if (
            self.feature_kd_weight > 0
            and student_features is not None
            and teacher_features is not None
        ):
            sf = F.normalize(student_features[valid], dim=1)
            tf = F.normalize(teacher_features[valid], dim=1)
            loss_feature = 0.5 * F.l1_loss(sf, tf) + 0.5 * (1 - F.cosine_similarity(sf, tf, dim=1).mean())

        total = self.alpha * loss_soft + (1 - self.alpha) * loss_hard + self.feature_kd_weight * loss_feature
        return {"total": total, "hard": loss_hard, "soft": loss_soft, "feature": loss_feature}


def load_labels(mark_version):
    cand_h = [
        os.path.join(PROJECT_ROOT, "extraction", f"hard_labels_{mark_version}.pkl"),
        os.path.join(PROJECT_ROOT, f"hard_labels_{mark_version}.pkl"),
        os.path.join(PROJECT_ROOT, "model", f"hard_labels_{mark_version}.pkl"),
    ]
    cand_s = [
        os.path.join(PROJECT_ROOT, "extraction", f"soft_labels_{mark_version}.pkl"),
        os.path.join(PROJECT_ROOT, f"soft_labels_{mark_version}.pkl"),
        os.path.join(PROJECT_ROOT, "model", f"soft_labels_{mark_version}.pkl"),
    ]
    hp = next((p for p in cand_h if os.path.exists(p)), None)
    sp = next((p for p in cand_s if os.path.exists(p)), None)
    if hp is None or sp is None:
        raise FileNotFoundError("hard/soft labels not found. Run extraction first.")

    with open(hp, "rb") as f:
        hard = pickle.load(f)
    with open(sp, "rb") as f:
        soft = pickle.load(f)

    smap = {
        e["path"]: {
            "soft_labels": e["soft_labels"],
            "teacher_features": e.get("teacher_features"),
        }
        for e in soft
    }

    samples = []
    for e in hard:
        path = e["path"]
        if path not in smap:
            continue
        soft_entry = smap[path]
        h = torch.tensor(e["hard_labels"], dtype=torch.long)
        s = torch.tensor(soft_entry["soft_labels"], dtype=torch.float)
        tf = soft_entry["teacher_features"]
        teacher_features = torch.tensor(tf, dtype=torch.float) if tf is not None else None

        if len(h) != len(s):
            print(f"[Warn] length mismatch: {path}")
            continue
        if teacher_features is not None and len(h) != len(teacher_features):
            print(f"[Warn] feature length mismatch: {path}")
            continue
        samples.append((path, h, s, teacher_features))
    return samples


def _in_split(path: str, split: str) -> bool:
    p = path.replace("\\", "/")
    return f"/data/{split}/" in p


def collate_fn(batch, parser: AudioParser, embedding_dim: int, num_channels: int, n_mels: int, segment_length: int):
    mel_list, hard_list, soft_list, feat_list = [], [], [], []
    for path, h, s, teacher_features in batch:
        segs = parser.load_and_segment(path)
        if not segs:
            continue
        lengths = [len(segs), len(h), len(s)]
        if teacher_features is not None:
            lengths.append(len(teacher_features))
        k = min(lengths)
        if k == 0:
            continue

        mel = torch.stack(segs[:k])
        mel_list.append(mel)
        hard_list.append(h[:k])
        soft_list.append(s[:k])
        if teacher_features is None:
            feat_list.append(torch.zeros((k, embedding_dim), dtype=torch.float))
        else:
            feat_list.append(teacher_features[:k])

    if not mel_list:
        return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)

    max_k = max(m.shape[0] for m in mel_list)
    num_c = soft_list[0].shape[1]
    for i in range(len(mel_list)):
        cur = mel_list[i].shape[0]
        if cur < max_k:
            mel_list[i] = torch.cat(
                [mel_list[i], torch.zeros((max_k - cur, num_channels, n_mels, segment_length))],
                dim=0,
            )
            hard_list[i] = torch.cat([hard_list[i], torch.full((max_k - cur,), -1, dtype=torch.long)], dim=0)
            soft_list[i] = torch.cat([soft_list[i], torch.zeros((max_k - cur, num_c))], dim=0)
            feat_list[i] = torch.cat([feat_list[i], torch.zeros((max_k - cur, embedding_dim))], dim=0)

    return torch.stack(mel_list), torch.stack(hard_list), torch.stack(soft_list), torch.stack(feat_list)


def train_student_with_distillation(seed_value=42, mark_version="mark4.1"):
    set_seed(seed_value)
    config = AudioViLDConfig(mark_version=mark_version)
    device = torch.device(config.device)
    parser = AudioParser(config)

    samples = load_labels(mark_version)
    train_data = [s for s in samples if _in_split(s[0], "train")]
    val_data = [s for s in samples if _in_split(s[0], "val")]

    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(
            b,
            parser,
            config.embedding_dim,
            config.num_input_channels,
            config.n_mels,
            config.segment_length,
        ),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(
            b,
            parser,
            config.embedding_dim,
            config.num_input_channels,
            config.n_mels,
            config.segment_length,
        ),
    )

    encoder = SimpleAudioEncoder(config).to(device)
    branch_head = DualBranchStudentHead(config.embedding_dim).to(device)
    text_head = ViLDTextHead(config).to(device)
    text_emb = config.get_class_text_embeddings().to(device)

    background_embedding = None
    if config.use_background_embedding:
        background_embedding = LearnableBackgroundEmbedding(config.embedding_dim).to(device)

    T = 4.0
    alpha = 0.7
    crit = DistillationLoss(
        T=T,
        alpha=alpha,
        feature_kd_weight=config.feature_kd_weight if config.use_feature_kd else 0.0,
        ignore_index=-1,
    )
    opt_params = list(encoder.parameters()) + list(branch_head.parameters())
    if background_embedding is not None:
        opt_params += list(background_embedding.parameters())
    opt = optim.Adam(
        opt_params,
        lr=config.learning_rate,
    )
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)

    enc_path = os.path.join(SCRIPT_DIR, f"distilled_student_encoder_{config.mark_version}.pth")
    head_path = os.path.join(SCRIPT_DIR, f"distilled_student_head_{config.mark_version}.pth")
    stopper = EarlyStopping(
        patience=10,
        verbose=True,
        path_encoder=enc_path,
        path_head=head_path,
        mark_version=config.mark_version,
    )

    tr_hist, vl_hist = [], []
    print(f"[INFO] Student KD training for {mark_version} on {device}")

    for ep in range(config.num_epochs):
        encoder.train()
        branch_head.train()
        total = total_h = total_s = total_f = 0.0
        for mb, hb, sb, fb in tqdm(train_loader, desc=f"[Train {ep+1}]"):
            if mb.numel() == 0:
                continue
            B, K, C, H, W = mb.shape
            mb = mb.view(B * K, C, H, W).to(device)
            hb = hb.view(B * K).to(device)
            sb = sb.view(B * K, -1).to(device)
            fb = fb.view(B * K, -1).to(device)

            base_features = encoder(mb)
            supervised_features, distill_features = branch_head(base_features)
            logits = text_head(supervised_features, text_emb)
            losses = crit(logits, sb, hb, student_features=distill_features, teacher_features=fb)

            if config.use_background_embedding and background_embedding is not None:
                others_idx = config.get_classes_for_text_prompts().index("others")
                valid = hb != -1
                others_mask = valid & (hb == others_idx)
                if others_mask.any():
                    target_feat = F.normalize(supervised_features[others_mask], dim=1)
                    bg = F.normalize(background_embedding(), dim=0).unsqueeze(0).expand_as(target_feat)
                    bg_loss = (1 - F.cosine_similarity(target_feat, bg, dim=1)).mean()
                    losses["total"] = losses["total"] + config.background_embedding_weight * bg_loss

            opt.zero_grad()
            losses["total"].backward()
            opt.step()

            total += losses["total"].item()
            total_h += losses["hard"].item()
            total_s += losses["soft"].item()
            total_f += losses["feature"].item()
        tr = total / max(1, len(train_loader))
        tr_hist.append(tr)

        encoder.eval()
        branch_head.eval()
        total = total_h = total_s = total_f = 0.0
        with torch.no_grad():
            for mb, hb, sb, fb in val_loader:
                if mb.numel() == 0:
                    continue
                B, K, C, H, W = mb.shape
                mb = mb.view(B * K, C, H, W).to(device)
                hb = hb.view(B * K).to(device)
                sb = sb.view(B * K, -1).to(device)
                fb = fb.view(B * K, -1).to(device)

                base_features = encoder(mb)
                supervised_features, distill_features = branch_head(base_features)
                logits = text_head(supervised_features, text_emb)
                losses = crit(logits, sb, hb, student_features=distill_features, teacher_features=fb)
                total += losses["total"].item()
                total_h += losses["hard"].item()
                total_s += losses["soft"].item()
                total_f += losses["feature"].item()
        vl = total / max(1, len(val_loader))
        vl_hist.append(vl)

        print(
            f"\n[Epoch {ep+1}] Train {tr:.6f} | Val {vl:.6f} | "
            f"Hard {total_h / max(1, len(val_loader)):.6f} | "
            f"Soft {total_s / max(1, len(val_loader)):.6f} | "
            f"Feat {total_f / max(1, len(val_loader)):.6f}"
        )

        stopper(vl, encoder, branch_head, background_embedding)
        if stopper.early_stop:
            print("[INFO] Early stopping.")
            break

        prev = opt.param_groups[0]["lr"]
        sched.step(vl)
        new = opt.param_groups[0]["lr"]
        if new < prev:
            print(f"[LR] {prev:.6g} -> {new:.6g} (val={vl:.6f})")

    plots = os.path.join(PROJECT_ROOT, "plots")
    os.makedirs(plots, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(tr_hist, label="Train")
    plt.plot(vl_hist, label="Val")
    plt.title(f"Distilled Student Loss ({mark_version})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out = os.path.join(plots, f"loss_curve_distilled_student_{mark_version}.png")
    plt.savefig(out)
    print(f"[INFO] Saved loss curve: {out}")

    # [추가] 손실곡선 raw 숫자를 CSV로도 저장. PNG만 있으면 다른 모델(CED-Tiny 등)과
    # 겹쳐 그리는 비교 그래프를 다시 만들 수 없어서, 비교 실험용으로 숫자 그대로 남긴다.
    loss_history_csv = os.path.join(plots, f"loss_history_{mark_version}.csv")
    with open(loss_history_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_total", "val_total"])
        for i in range(len(tr_hist)):
            writer.writerow([i + 1, tr_hist[i], vl_hist[i]])
    print(f"[INFO] 손실곡선 CSV 저장: {loss_history_csv}")
    print(f"[INFO] Best saved: {enc_path}, {head_path} (val loss {stopper.val_loss_min:.6f})")
    save_checkpoint(
        os.path.join(SCRIPT_DIR, f"student_checkpoint_{config.mark_version}.pt"),
        model_type="student_full",
        mark_version=config.mark_version,
        model_state=encoder.state_dict(),
        branch_state=branch_head.state_dict(),
        classifier_state=text_head.state_dict(),
        background_embedding_state=(
            background_embedding.state_dict()
            if (config.use_background_embedding and background_embedding is not None)
            else None
        ),
    )


def train_student(seed_value=42, mark_version="mark4.1"):
    return train_student_with_distillation(seed_value=seed_value, mark_version=mark_version)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mark_version", type=str, required=True)
    args = ap.parse_args()
    train_student_with_distillation(mark_version=args.mark_version)
