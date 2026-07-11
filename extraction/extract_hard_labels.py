# extraction/extract_hard_labels.py

"""
[Deprecated: train/val/test 전체에서 하드라벨을 생성하던 방식]
# all rows -> hard_labels
"""
import os, sys, pickle, csv
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))        # extraction/
PROJECT_ROOT = os.path.dirname(BASE_DIR)
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
VILD_DIR = os.path.join(PROJECT_ROOT, "vild")
for p in (PROJECT_ROOT, UTILS_DIR, VILD_DIR):
    if p not in sys.path: sys.path.append(p)

from vild_config import AudioViLDConfig
from autoNor_utils import normalize_label
from vild_parser_student import AudioParser  # 세그먼트 수 확보용

def _resolve_csv_path(mark_version: str) -> str:
    candidates = [
        os.path.join(PROJECT_ROOT, f"dataset_index_{mark_version}.csv"),
        os.path.join(BASE_DIR, f"dataset_index_{mark_version}.csv"),
        os.path.join(PROJECT_ROOT, "preprocessing", f"dataset_index_{mark_version}.csv"),
        os.path.join(PROJECT_ROOT, "model", f"dataset_index_{mark_version}.csv"),
    ]
    for c in candidates:
        if os.path.exists(c): return c
    raise FileNotFoundError(f"CSV not found: {candidates}")

def _in_split(path: str, allowed={"train", "val"}) -> bool:
    p = path.replace("\\", "/")
    return any(f"/data/{s}/" in p for s in allowed)

def _mirror_save(obj, filename: str):
    targets = [
        os.path.join(PROJECT_ROOT, filename),
        os.path.join(BASE_DIR, filename),
        os.path.join(PROJECT_ROOT, "model", filename),
    ]
    for t in targets:
        os.makedirs(os.path.dirname(t), exist_ok=True)
        with open(t, "wb") as f: pickle.dump(obj, f)
        print(f"[INFO] Saved: {t}")

def extract_hard_labels(mark_version="mark4.1"):
    config = AudioViLDConfig(mark_version=mark_version)
    csv_path = _resolve_csv_path(mark_version)
    parser = AudioParser(config)

    hard_label_entries = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in tqdm(list(csv.DictReader(f)), desc=f"Hard labels ({mark_version})"):
            path = row['path']; label = normalize_label(row['label'])
            if not _in_split(path, {"train","val"}):  # test 제외
                continue
            try:
                label_idx = config.get_class_index(label)
            except ValueError:
                print(f"[Warn] Unknown label: {label} ({path})"); continue

            segs = parser.load_and_segment(path)
            if not segs:
                print(f"[Warn] No segments: {path}"); continue

            hard_label_entries.append({
                "path": path,
                "hard_labels": [label_idx]*len(segs)
            })

    out_name = f"hard_labels_{mark_version}.pkl"
    _mirror_save(hard_label_entries, out_name)
    print(f"[완료] hard_labels(train+val) 저장: {len(hard_label_entries)} files")
    return 0

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--mark_version', type=str, required=True)
    args = ap.parse_args()
    raise SystemExit(extract_hard_labels(args.mark_version))
    