# model/teacher_train.py

"""
[Deprecated: dataset_index 전체를 random_split(80/20)으로 나누던 방식]
# full_dataset = LabeledAudioDataset(file_label_list, parser, config)
# val_size = max(1, int(0.2 * len(full_dataset)))
# train_size = len(full_dataset) - val_size
# train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], ...)
"""
import os
import sys
import csv
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))            # model/
PROJECT_ROOT = os.path.dirname(BASE_DIR)                         # mark4.x/
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
VILD_DIR = os.path.join(PROJECT_ROOT, "vild")
for p in (PROJECT_ROOT, UTILS_DIR, VILD_DIR):
    if p not in sys.path:
        sys.path.append(p)

from vild_config import AudioViLDConfig
from vild_model import SimpleAudioEncoder, ViLDTextHead, build_teacher_encoder
from vild_parser_teacher import AudioParser
from vild_losses import ViLDLosses
from seed_utils import set_seed
SHARED_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "shared_vild"))
if SHARED_DIR not in sys.path:
    sys.path.append(SHARED_DIR)
from checkpoint_utils import save_checkpoint

def _resolve_csv_path(mark_version: str) -> str:
    candidates = [
        os.path.join(PROJECT_ROOT, f"dataset_index_{mark_version}.csv"),
        os.path.join(BASE_DIR, f"dataset_index_{mark_version}.csv"),
        os.path.join(PROJECT_ROOT, "preprocessing", f"dataset_index_{mark_version}.csv"),
        os.path.join(PROJECT_ROOT, "extraction", f"dataset_index_{mark_version}.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"[ERROR] dataset_index CSV not found for {mark_version}")

def _in_split(path: str, split: str) -> bool:
    p = path.replace("\\", "/")  # 현재 in_split은 ‘/data/{split}/’ 외에 ‘/data/{split}’도 허용 -> ‘/data/train_heavy_impact/’ 같은 확장 폴더가 있어도 train으로 포함가능.
    return f"/data/{split}/" in p or p.endswith(f"/data/{split}") or f"/data/{split}_" in p

class LabeledAudioDataset(Dataset):
    """
    - parser.load_and_segment 사용
    - 각 세그먼트를 개별 샘플로 저장 (seg, label)
    """
    def __init__(self, file_label_list, parser, config):
        self.samples = []
        self.parser = parser
        self.config = config
        valid_labels = set(config.get_classes_for_text_prompts())

        for path, label in file_label_list:
            if label in valid_labels:
                try:
                    segments = self.parser.load_and_segment(path)
                    for seg in segments:
                        if seg is not None:
                            self.samples.append((seg, label))
                except Exception as e:
                    print(f"[ERROR] Failed to parse {path}: {e}")

        if not self.samples:
            print("[Warning] No valid audio segments found in dataset.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def custom_collate(batch):
    mels, labels = zip(*batch)
    mels = torch.stack(mels, dim=0)
    return mels, labels

def _spec_augment(mel, freq_mask_param=12, time_mask_param=20):
    """
    [추가 2026-07-13 / teacher 강화] SpecAugment(주파수/시간 마스킹) — teacher 학습 배치 전용.
    샘플마다 주파수 밴드 1개 + 시간 구간 1개를 무작위 폭으로 가린다. 마스킹 값은 해당 샘플의
    최솟값(dB 스펙트로그램의 최저 = 무음)이라 '0으로 채우면 중간 에너지가 되어버리는' 함정을 피함.
    val/eval 경로에는 적용하지 않는다(clean 평가 유지).
    """
    out = mel.clone()
    B, _, n_mels, n_time = out.shape
    for i in range(B):
        fill = out[i].min()
        f = int(torch.randint(0, freq_mask_param + 1, (1,)).item())
        if f > 0 and n_mels > f:
            f0 = int(torch.randint(0, n_mels - f, (1,)).item())
            out[i, :, f0:f0 + f, :] = fill
        t = int(torch.randint(0, time_mask_param + 1, (1,)).item())
        if t > 0 and n_time > t:
            t0 = int(torch.randint(0, n_time - t, (1,)).item())
            out[i, :, :, t0:t0 + t] = fill
    return out

def train_teacher(seed_value=42, mark_version="mark4.1"):
    set_seed(seed_value)
    config = AudioViLDConfig(mark_version=mark_version)
    parser = AudioParser(config, segment_mode=True)
    device = config.device

    csv_path = _resolve_csv_path(mark_version)
    all_list = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_list.append((row["path"], row["label"]))

    # 학습/검증을 파일 경로 split으로 엄격 분리
    train_list = [(p, l) for (p, l) in all_list if _in_split(p, "train")]
    val_list   = [(p, l) for (p, l) in all_list if _in_split(p, "val")]

    train_dataset = LabeledAudioDataset(train_list, parser, config)
    val_dataset   = LabeledAudioDataset(val_list, parser, config)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,  collate_fn=custom_collate)
    val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False, collate_fn=custom_collate)

    # [수정 2026-07-13 / teacher 강화] student와 같은 초경량 인코더 대신 teacher 전용 대형
    # 인코더 사용(config.use_large_teacher_encoder). extract_soft_labels.py도 같은 팩토리를
    # 쓰므로 체크포인트 구조가 항상 일치한다.
    teacher_encoder = build_teacher_encoder(config).to(device)
    teacher_classifier = ViLDTextHead(config).to(device)
    n_params = sum(p.numel() for p in teacher_encoder.parameters())
    print(f"[INFO] Teacher encoder: {type(teacher_encoder).__name__} ({n_params:,} params)")
    # [수정 2026-07-12 / 가설6] weight_decay 추가(기존 0). teacher가 val best를 epoch 5에
    # 찍고 곧장 과적합(train down/val up)하던 실측에 대한 L2 정규화 처방.
    optimizer = optim.Adam(list(teacher_encoder.parameters()) + list(teacher_classifier.parameters()),
                           lr=config.learning_rate,
                           weight_decay=getattr(config, "weight_decay", 0.0))
    # [추가 2026-07-13 / teacher 강화] LR 스케줄(student에는 이미 있고 teacher에는 없었음).
    # 고정 LR이 val 요동(0.52~0.62)의 한 원인 -> val loss 정체 시 절반으로.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=getattr(config, "teacher_lr_factor", 0.5),
        patience=getattr(config, "teacher_lr_patience", 3),
    )
    use_specaug = getattr(config, "teacher_spec_augment", False)
    loss_fn = ViLDLosses(config)

    print(f"[INFO] Teacher training started for {mark_version} on {device}"
          + (f" | SpecAugment on (F<={getattr(config,'teacher_freq_mask_param',12)},"
             f" T<={getattr(config,'teacher_time_mask_param',20)})" if use_specaug else ""))
    best_val = float('inf')
    # [수정 2026-07-11] patience=2 하드코딩 -> config.teacher_patience(기본 10)로 변경.
    # num_epochs=80까지 돌 수 있는데 val loss가 한 번만 반등해도 5epoch 안에 조기종료되어,
    # teacher가 제대로 수렴하기 전에 학습이 끝나버리는 버그였음(실측: mark4.8이 5epoch에서 멈춤).
    patience, wait = getattr(config, "teacher_patience", 10), 0
    train_hist, val_hist = [], []

    text_emb = config.get_class_text_embeddings().to(device)
    label_map = config.get_target_label_map()
    for epoch in range(config.num_epochs):
        teacher_encoder.train(); teacher_classifier.train()
        total = 0.0
        for mel_batch, label_batch in train_loader:
            mel = mel_batch.to(device)
            if mel.dim() == 5:
                mel = mel.squeeze(1)
            elif mel.dim() == 3:
                mel = mel.unsqueeze(1)

            if use_specaug:
                mel = _spec_augment(
                    mel,
                    freq_mask_param=getattr(config, "teacher_freq_mask_param", 12),
                    time_mask_param=getattr(config, "teacher_time_mask_param", 20),
                )

            targets = torch.tensor([label_map[l] for l in label_batch], dtype=torch.long).to(device)
            region = teacher_encoder(mel)
            logits = teacher_classifier(region, text_emb)
            loss = loss_fn.compute_text_loss(logits, targets)

            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item()
        tr = total / max(1, len(train_loader)); train_hist.append(tr)

        teacher_encoder.eval(); teacher_classifier.eval()
        total = 0.0
        with torch.no_grad():
            for mel_batch, label_batch in val_loader:
                mel = mel_batch.to(device)
                if mel.dim() == 5:
                    mel = mel.squeeze(1)
                elif mel.dim() == 3:
                    mel = mel.unsqueeze(1)

                targets = torch.tensor([label_map[l] for l in label_batch], dtype=torch.long).to(device)
                region = teacher_encoder(mel)
                logits = teacher_classifier(region, text_emb)
                loss = loss_fn.compute_text_loss(logits, targets)
                total += loss.item()
        vl = total / max(1, len(val_loader)); val_hist.append(vl)
        print(f"[Epoch {epoch+1}] Train {tr:.4f} | Val {vl:.4f}")

        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(vl)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != prev_lr:
            print(f"[LR] {prev_lr:g} -> {new_lr:g} (val={vl:.6f})")

        if vl < best_val:
            best_val = vl; wait = 0
            save_checkpoint(
                os.path.join(BASE_DIR, f"best_teacher_encoder_{mark_version}.pth"),
                model_type="teacher_encoder",
                mark_version=mark_version,
                model_state=teacher_encoder.state_dict(),
                text_embeddings=text_emb.detach().cpu(),
                config=config,
            )
            save_checkpoint(
                os.path.join(BASE_DIR, f"best_teacher_classifier_{mark_version}.pth"),
                model_type="teacher_classifier",
                mark_version=mark_version,
                classifier_state=teacher_classifier.state_dict(),
                text_embeddings=text_emb.detach().cpu(),
                config=config,
            )
            print("[INFO] Improved. Saved best teacher.")
        else:
            wait += 1
            if wait >= patience:
                print("[INFO] Early stopping."); break

    # 최종 체크포인트
    save_checkpoint(
        os.path.join(BASE_DIR, f"teacher_checkpoint_{mark_version}.pt"),
        model_type="teacher_full",
        mark_version=mark_version,
        model_state=teacher_encoder.state_dict(),
        classifier_state=teacher_classifier.state_dict(),
        text_embeddings=text_emb.detach().cpu(),
        config=config,
    )

    # 손실 그래프
    plots = os.path.join(PROJECT_ROOT, "plots"); os.makedirs(plots, exist_ok=True)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5)); plt.plot(train_hist, label='Train'); plt.plot(val_hist, label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Teacher Loss'); plt.legend(); plt.grid(True); plt.tight_layout()
    out_png = os.path.join(plots, f"loss_curve_teacher_train_val_{mark_version}.png")
    plt.savefig(out_png); print("[INFO] Loss curve saved:", out_png)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mark_version", type=str, required=True)
    args = parser.parse_args()
    train_teacher(seed_value=42, mark_version=args.mark_version)
    
