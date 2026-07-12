"""
augment_density.py — 클래스간 '소리 밀도' 격차를 좁히는 증강(4.x 시리즈 공용).

배경(2026-07-12): mark4.8 데이터에서 target 클래스(예: dog_bark)는 소리가 성기고
others는 빽빽한 계통 격차가 있었음(active 비율 dog 0.52 vs others 0.67, split마다 일관).
salient_topk 세그먼트 선택으로도 상쇄되지 않아(격차 -0.150 -> -0.155) 모델이 음색 대신
밀도를 지름길로 학습할 위험이 있었음. 이 스크립트는 target 클래스를 '빽빽하게'(여러 시점에
짖음을 겹침 = 반복해 짖는 개), others를 '띄엄띄엄하게'(무음 구간 삽입) 증강해 두 클래스의
밀도 분포가 겹치게 만든다. 목적은 밀도를 지우는 게 아니라 모델이 소리의 전 범위(성김+빽빽)에
강건해지게 하는 것.

정책:
- 기본적으로 train split에만 적용(val/test는 실제 분포 유지 — data leakage 방지).
- 원본은 건드리지 않고 새 파일(_aug_NNN)로 추가 저장.
- data_provenance.xlsx에 증강본 행 추가(source_type/aug_method/aug_source_file/active_ratio 기록).
- 오디오 길이는 원본과 동일(3초=48000샘플 @16kHz)로 유지 -> fix_audio_length가 다시 안 건드림.

사용 예:
  python preprocessing/augment_density.py --mark_version mark4.8
  python preprocessing/augment_density.py --mark_version mark4.8 --target_split train --n_aug_per_class 100 --seed 42
"""
import os
import sys
import glob
import time
import hashlib
import argparse

import numpy as np
import soundfile as sf


# ===== 활성(소리) 구간 측정 유틸 =====
def _frame_rms(x, sr):
    fr = int(0.025 * sr); fhop = int(0.010 * sr)
    if len(x) < fr:
        return np.array([np.sqrt(np.mean(x ** 2) + 1e-12)])
    n = 1 + (len(x) - fr) // fhop
    return np.array([np.sqrt(np.mean(x[i * fhop:i * fhop + fr] ** 2) + 1e-12) for i in range(n)])


def active_ratio(x, sr, abs_floor=1e-4, rel=0.05):
    """소리 있는 프레임 비율(0~1). 무음(패딩)은 제외."""
    rms = _frame_rms(x, sr)
    if len(rms) == 0:
        return 0.0
    thr = max(abs_floor, rel * rms.max())
    return float(np.mean(rms > thr))


# ===== 증강 변환 =====
def densify(x, sr, seed_rng, all_shifts=(0.4, 0.8, 1.2, 1.6, 2.0)):
    """시간이동 겹침으로 빽빽하게(왈 왈 왈 — 반복해 짖는 개). 파일마다 겹침 횟수를 1~5로
    랜덤하게 골라 중간~빽빽 밀도를 골고루 커버(한쪽으로 과도하게 쏠리는 것 방지).
    길이 유지, 원본 피크로 정규화."""
    peak = float(np.max(np.abs(x)))
    if peak < 1e-4:
        return x.copy()
    k = int(seed_rng.integers(1, len(all_shifts) + 1))       # 1~5회
    shifts = seed_rng.choice(all_shifts, size=k, replace=False)
    out = x.astype(np.float64).copy()
    for d in shifts:
        out = out + np.roll(x, int(float(d) * sr))
    m = float(np.max(np.abs(out)))
    if m > 0:
        out = out / m * min(peak, 0.99)
    return out


def sparsify(x, sr, target_active, seed_rng, chunk_sec=0.3):
    """무음 구간을 넣어 띄엄띄엄하게. target_active 비율 근처까지 청크를 무음화."""
    out = x.astype(np.float64).copy()
    chunk = int(chunk_sec * sr)
    n_chunks = max(1, len(out) // chunk)
    order = seed_rng.permutation(n_chunks)
    for c in order:
        if active_ratio(out, sr) <= target_active:
            break
        out[c * chunk:(c + 1) * chunk] = 0.0
    return out


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="클래스간 소리 밀도 격차 완화 증강(train 한정 기본)")
    parser.add_argument("--mark_version", type=str, required=True, help="예: mark4.8")
    parser.add_argument("--target_split", type=str, default="train", help="증강 대상 split(기본 train)")
    parser.add_argument("--n_aug_per_class", type=int, default=100, help="클래스당 생성할 증강본 수")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--provenance_path", type=str, default=None,
                        help="data_provenance.xlsx 경로(기본: codes/data_provenance.xlsx)")
    parser.add_argument("--dry_run", action="store_true", help="파일/엑셀 안 쓰고 계획만 출력")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    split_dir = os.path.join(project_root, "data", args.target_split)
    if not os.path.isdir(split_dir):
        print(f"[ERROR] split 폴더 없음: {split_dir}")
        sys.exit(1)

    if args.provenance_path:
        prov_path = args.provenance_path
    else:
        # codes/ 루트(= project_root의 상위)에 data_provenance.xlsx
        prov_path = os.path.join(os.path.dirname(project_root), "data_provenance.xlsx")

    # ---- 파일 로드 + 클래스 판별(파일명 접두 target_class로) ----
    wavs = sorted(glob.glob(os.path.join(split_dir, "*.wav")))
    # 증강본(_aug)은 소스에서 제외(재실행 시 중복 증강 방지)
    wavs = [w for w in wavs if "_aug_" not in os.path.basename(w)]
    if not wavs:
        print(f"[ERROR] {split_dir}에 원본 wav 없음")
        sys.exit(1)

    by_class = {}
    for w in wavs:
        base = os.path.basename(w)
        cls = base.rsplit(f"_{args.target_split}_", 1)[0]  # dog_bark / others
        x, sr = sf.read(w)
        if x.ndim > 1:
            x = x.mean(axis=1)
        by_class.setdefault(cls, []).append((w, x.astype(np.float64), sr, active_ratio(x, sr)))

    if len(by_class) != 2:
        print(f"[WARN] 클래스가 2개가 아님({list(by_class)}). 2-class specialist 전제 스크립트입니다.")

    sr0 = by_class[next(iter(by_class))][0][2]
    means = {c: float(np.mean([a for _, _, _, a in items])) for c, items in by_class.items()}
    sparse_cls = min(means, key=means.get)   # 밀도 낮은 클래스 -> 빽빽화
    dense_cls = max(means, key=means.get)    # 밀도 높은 클래스 -> 성김화
    print(f"[대상] split={args.target_split}  클래스별 active 평균: " +
          ", ".join(f"{c}={m:.4f}" for c, m in means.items()))
    print(f"  성긴 클래스(빽빽화 대상): {sparse_cls}  /  빽빽한 클래스(성김화 대상): {dense_cls}")
    sparse_target = means[sparse_cls]  # 빽빽한 쪽을 성긴 쪽 수준으로 낮춤

    # ---- 증강 생성 ----
    plan = []  # (out_path, waveform, method, source_base, active)
    # 성긴 클래스: 성긴 것부터 골라 densify
    src_sparse = sorted(by_class[sparse_cls], key=lambda t: t[3])
    for i in range(args.n_aug_per_class):
        srcpath, x, sr, _ = src_sparse[i % len(src_sparse)]
        aug = densify(x, sr, rng)
        out_base = f"{sparse_cls}_{args.target_split}_aug_{i+1:03d}.wav"
        plan.append((os.path.join(split_dir, out_base), aug, "densify", os.path.basename(srcpath), active_ratio(aug, sr)))
    # 빽빽한 클래스: 빽빽한 것부터 골라 sparsify
    src_dense = sorted(by_class[dense_cls], key=lambda t: t[3], reverse=True)
    for i in range(args.n_aug_per_class):
        srcpath, x, sr, _ = src_dense[i % len(src_dense)]
        aug = sparsify(x, sr, sparse_target, rng)
        out_base = f"{dense_cls}_{args.target_split}_aug_{i+1:03d}.wav"
        plan.append((os.path.join(split_dir, out_base), aug, "sparsify", os.path.basename(srcpath), active_ratio(aug, sr)))

    # ---- 증강 후 예상 분포 ----
    def all_active(cls):
        orig = [a for _, _, _, a in by_class[cls]]
        aug = [p[4] for p in plan if os.path.basename(p[0]).startswith(cls + f"_{args.target_split}_aug_")]
        return orig + aug
    da = all_active(sparse_cls); oa = all_active(dense_cls)
    print(f"[증강 후 예상] {sparse_cls} mean={np.mean(da):.4f}  {dense_cls} mean={np.mean(oa):.4f}  격차={np.mean(da)-np.mean(oa):+.4f}")
    print(f"  생성 예정 파일: {len(plan)}개 (클래스당 {args.n_aug_per_class})")

    if args.dry_run:
        print("[dry_run] 파일/엑셀 안 씀. 계획만 출력하고 종료.")
        return

    # ---- 파일 쓰기 ----
    for out_path, wav, method, src_base, act in plan:
        sf.write(out_path, wav.astype(np.float32), sr0, subtype="FLOAT")
    print(f"[완료] 증강 wav {len(plan)}개 저장 -> {split_dir}")

    # ---- provenance 갱신 ----
    import pandas as pd
    df = pd.read_excel(prov_path)
    # 증강 기록용 컬럼(없으면 추가, 기존 행은 original로 채움)
    if "source_type" not in df.columns:
        df["source_type"] = "original"
    if "aug_method" not in df.columns:
        df["aug_method"] = ""
    if "aug_source_file" not in df.columns:
        df["aug_source_file"] = ""
    if "active_ratio" not in df.columns:
        df["active_ratio"] = np.nan

    today = time.strftime("%Y-%m-%d")
    new_rows = []
    for out_path, wav, method, src_base, act in plan:
        base = os.path.basename(out_path)
        cls = base.rsplit(f"_{args.target_split}_", 1)[0]
        # 소스 원본 행에서 라벨 등 상속
        src_row = df[df["local_filename"] == src_base]
        orig_labels = src_row["original_labels"].iloc[0] if len(src_row) else ""
        new_rows.append({
            "local_filename": base,
            "fsd50k_fname": "",
            "fsd50k_split": "augmented",
            "original_labels": orig_labels,
            "target_class": cls,
            "assigned_split": args.target_split,
            "mark_version": args.mark_version,
            "sha256": _sha256(out_path),
            "source_volume": "augmented",
            "size_bytes": os.path.getsize(out_path),
            "download_date": today,
            "source_type": "augmented",
            "aug_method": method,
            "aug_source_file": src_base,
            "active_ratio": round(act, 4),
        })
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_excel(prov_path, index=False)
    print(f"[완료] data_provenance.xlsx에 증강본 {len(new_rows)}행 추가 (총 {len(df)}행) -> {prov_path}")
    print("[다음] generate_dataset_index.py 재실행 또는 run_all.py로 인덱스 갱신 필요.")


if __name__ == "__main__":
    main()
