# generate_dataset_index.py

import os
import sys
import glob
import csv
import pickle
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt

# === 경로 설정 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# [변경] utils/vild 모듈 접근을 위해 프로젝트 루트 기준 경로를 추가
PROJECT_ROOT = os.path.dirname(BASE_DIR)
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
VILD_DIR = os.path.join(PROJECT_ROOT, "vild")
for p in (PROJECT_ROOT, UTILS_DIR, VILD_DIR):
    if p not in sys.path:
        sys.path.append(p)

from seed_utils import set_seed
from autoNor_utils import normalize_label
from vild_config import AudioViLDConfig


def get_filename_keyword_map_for_version(config: AudioViLDConfig):
    mark_version = config.mark_version
    labeled = config.labeled_classes
    unlabeled = config.unlabeled_class_identifier

    keyword_map = {  # check needed
        "mark4.1": {
            "heavy_impact": labeled[0],
            "others": labeled[1],
            "unlabeled": unlabeled
        },
        "mark4.2": {
            "dragging": labeled[0],
            "others": labeled[1],
            "unlabeled": unlabeled
        },
        "mark4.3": {
            "construction": labeled[0],
            "others": labeled[1],
            "unlabeled": unlabeled
        },
        "mark4.4": {
            "machine_noise": labeled[0],
            "others": labeled[1],
            "unlabeled": unlabeled
        },
        "mark4.5": {
            "media_talking": labeled[0],
            "others": labeled[1],
            "unlabeled": unlabeled
        },
        "mark4.6": {
            "water_toilet": labeled[0],
            "others": labeled[1],
            "unlabeled": unlabeled
        },
        "mark4.7": {
            "water_shower": labeled[0],
            "others": labeled[1],
            "unlabeled": unlabeled
        },
        "mark4.8": {
            "dog_bark": labeled[0],
            "others": labeled[1],
            "unlabeled": unlabeled
        }
    }

    if mark_version not in keyword_map:
        raise ValueError(f"No keyword map defined for mark_version '{mark_version}'")

    return keyword_map[mark_version]


def save_csv(entries, output_path):
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(entries)


def save_pkl(entries, output_path):
    data = [{"path": path, "label": label} for path, label in entries]
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)


def plot_label_distribution(label_count: dict, mark_version: str, save_dir: str = "plots"):
    os.makedirs(save_dir, exist_ok=True)
    labels = list(label_count.keys())
    counts = [label_count[label] for label in labels]

    plt.figure(figsize=(8, 4))
    plt.bar(labels, counts, color='skyblue')
    plt.title(f"Label Distribution - {mark_version}")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"label_dist_{mark_version}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[완료] 라벨 분포 시각화 저장됨: {save_path}")


def generate_index(mark_version: str, seed_value: int = 42):
    set_seed(seed_value)
    config = AudioViLDConfig(mark_version=mark_version)

    """
    [Deprecated: mark2.x 호환을 위한 data_wav 단일 폴더 스캔]
    current_data_dir = os.path.join(BASE_DIR, "data_wav")
    if not os.path.isdir(current_data_dir):
        raise FileNotFoundError(f"[ERROR] Data directory not found: {current_data_dir}")
    audio_paths = sorted(glob.glob(os.path.join(current_data_dir, '*.wav')))
    """

    # [변경] mark4.x 구조: data/{train,val,test} 각각 스캔
    data_root = os.path.join(PROJECT_ROOT, "data")
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"[ERROR] Data directory not found: {data_root}")

    splits = ["train", "val", "test"]
    split_paths = []
    for sp in splits:
        sp_dir = os.path.join(data_root, sp)
        if not os.path.isdir(sp_dir):
            print(f"[Warning] Split not found, skipped: {sp_dir}")
            continue
        split_paths.extend(sorted(glob.glob(os.path.join(sp_dir, "*.wav"))))

    if not split_paths:
        raise FileNotFoundError(f"[ERROR] No .wav files found under {data_root}/{{train,val,test}}")

    # 출력 파일은 프로젝트 루트에 저장하고, 호환성을 위해 model/, extraction/에도 복사
    output_csv_root = os.path.join(PROJECT_ROOT, f"dataset_index_{mark_version}.csv")
    output_pkl_root = os.path.join(PROJECT_ROOT, f"dataset_index_{mark_version}.pkl")

    filename_keyword_map = get_filename_keyword_map_for_version(config)

    print(f"\n[INFO] Checking data directory: {data_root}")
    print(f"--- Generating dataset index for: {mark_version} ---")
    print(f"  Data source directory: {data_root} (scanning train/val/test)")
    print(f"  Output CSV file: {os.path.basename(output_csv_root)}")
    print(f"  Output PKL file: {os.path.basename(output_pkl_root)}")

    entries = []
    found_labels_in_data = set()
    label_count = defaultdict(int)

    for path in split_paths:
        basename = os.path.basename(path).lower()
        matched_label = None

        for keyword in sorted(filename_keyword_map, key=len, reverse=True):
            if keyword in basename:
                label = filename_keyword_map[keyword]
                label = normalize_label(label)
                try:
                    _ = config.get_class_index(label)
                    matched_label = label
                    break
                except ValueError:
                    print(f"[Warning] Invalid label '{label}' in config.")

        if matched_label in config.labeled_classes:
            entries.append((path.replace("\\", "/"), matched_label))
            found_labels_in_data.add(matched_label)
            label_count[matched_label] += 1
        elif matched_label == config.unlabeled_class_identifier:
            continue
        else:
            print(f"[Notice] Skipping unrecognized file: {basename}")

    entries.sort(key=lambda x: x[0])

    # 저장(프로젝트 루트)
    save_csv(entries, output_csv_root)
    save_pkl(entries, output_pkl_root)

    # 호환성 유지: preprocessing/, model/, extraction/에도 동일 이름으로 복사 저장
    mirror_dirs = [
        BASE_DIR,
        os.path.join(PROJECT_ROOT, "model"),
        os.path.join(PROJECT_ROOT, "extraction"),
    ]
    for d in mirror_dirs:
        try:
            os.makedirs(d, exist_ok=True)
            save_csv(entries, os.path.join(d, f"dataset_index_{mark_version}.csv"))
            save_pkl(entries, os.path.join(d, f"dataset_index_{mark_version}.pkl"))
        except Exception as e:
            print(f"[Mirror Warning] Failed to mirror index to {d}: {e}")

    # 요약 출력
    print(f"\nSaved {len(entries)} entries to:")
    print(f"  - CSV: {output_csv_root}")
    print(f"  - PKL: {output_pkl_root}")

    print("\n--- Summary ---")
    expected_labels = set(config.labeled_classes)
    missing_labels = expected_labels - found_labels_in_data
    if missing_labels:
        print("[Warning] Missing expected labels:")
        for label in sorted(missing_labels):
            print(f"  - {label}")

    unexpected_labels = found_labels_in_data - expected_labels
    if unexpected_labels:
        print("[Warning] Unexpected labels found in data:")
        for label in sorted(unexpected_labels):
            print(f"  - {label}")

    print(f"\nTotal .wav files scanned: {len(split_paths)}")
    print(f"Total labeled entries written: {len(entries)}")
    print("\n[Label Distribution]")
    for label in sorted(label_count.keys()):
        print(f"  {label}: {label_count[label]} files")

    # 시각화 추가
    plots_dir = os.path.join(PROJECT_ROOT, "plots")
    plot_label_distribution(label_count, mark_version, save_dir=plots_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mark_version', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    generate_index(
        mark_version=args.mark_version,
        seed_value=args.seed
    )
    