# preprocessing/resplit_dataset.py
#
# [추가 2026-07-11] FSD50K dev/eval 경계를 무시하고 클래스별 파일을 무작위로
# train/val/test에 재배정한다.
#
# 배경: mark4.8 데이터 수집 시 target(dog_bark)/others 각각 200개를 FSD50K에서
# 받으면서, train+val(300개)을 전부 FSD50K "dev" split에서, test(100개)를 전부
# FSD50K "eval" split에서 뽑았다. FSD50K 공식 문서(Fonseca et al., 2020)에 따르면
# dev/eval은 업로더가 완전히 겹치지 않도록(disjoint uploaders) 의도적으로 분리된
# 도메인이라(eval은 라벨 수·클립 길이도 dev와 다름), 이 경계를 그대로 train-test
# 경계로 쓰면서 teacher가 train/val에서는 80%대 accuracy를 내면서도 test에서는
# 51%(2-class 랜덤 수준)로 붕괴했다. entropy_threshold 버그(vild_config.py)와는
# 별개의, 더 근본적인 데이터 분할 설계 문제였다.
#
# 이 스크립트는 이미 다운로드한 파일을 재수집하지 않고, data_provenance.xlsx에
# 기록된 클래스별 파일들을 fsd50k_split(dev/eval)과 무관하게 다시 무작위로 섞어
# 기존과 같은 비율(예: 100/50/50)로 train/val/test에 재배정한다. 재분할 후에는
# generate_dataset_index.py를 다시 돌려 dataset_index_{mark_version}.csv/pkl을
# 갱신해야 한다.
import argparse
import os
import random
import shutil
import sys

import pandas as pd

PREPROCESSING_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PREPROCESSING_DIR)
CODES_ROOT = os.path.dirname(PROJECT_ROOT)
VILD_DIR = os.path.join(PROJECT_ROOT, "vild")
if VILD_DIR not in sys.path:
    sys.path.append(VILD_DIR)

from vild_config import AudioViLDConfig  # noqa: E402


def resplit(mark_version: str, seed: int, provenance_path: str, split_sizes: list):
    random.seed(seed)

    config = AudioViLDConfig(mark_version=mark_version)
    target_classes = [c for c in config.get_classes_for_text_prompts() if c != "others"] + ["others"]

    data_root = os.path.join(PROJECT_ROOT, "data")
    staging = os.path.join(PROJECT_ROOT, f"_resplit_staging_{mark_version}")
    os.makedirs(staging, exist_ok=True)

    df = pd.read_excel(provenance_path)
    mask = df["mark_version"] == mark_version
    target_df = df[mask].copy()
    if target_df.empty:
        raise ValueError(f"[ERROR] data_provenance.xlsx에 mark_version='{mark_version}' 행이 없습니다.")

    new_rows = []
    for target_class, group in target_df.groupby("target_class"):
        rows = group.to_dict("records")
        expected = sum(n for _, n in split_sizes)
        if len(rows) != expected:
            raise ValueError(
                f"[ERROR] '{target_class}' 파일 수({len(rows)})가 split_sizes 합({expected})과 다릅니다."
            )
        random.shuffle(rows)

        idx = 0
        for split_name, count in split_sizes:
            for i in range(count):
                row = rows[idx]
                idx += 1
                old_path = os.path.join(data_root, row["assigned_split"], row["local_filename"])

                staged_path = os.path.join(staging, f"{target_class}_{row['fsd50k_fname']}.wav")
                if not os.path.exists(staged_path):
                    if not os.path.exists(old_path):
                        raise FileNotFoundError(f"원본 파일 없음: {old_path}")
                    shutil.move(old_path, staged_path)

                row["_staged_path"] = staged_path
                row["_new_split"] = split_name
                row["_new_filename"] = f"{target_class}_{split_name}_{i + 1:03d}.wav"
                new_rows.append(row)

    for row in new_rows:
        dst_dir = os.path.join(data_root, row["_new_split"])
        os.makedirs(dst_dir, exist_ok=True)
        shutil.move(row["_staged_path"], os.path.join(dst_dir, row["_new_filename"]))

    for row in new_rows:
        ridx = df[
            (df["mark_version"] == mark_version)
            & (df["fsd50k_fname"] == row["fsd50k_fname"])
            & (df["target_class"] == row["target_class"])
        ].index
        if len(ridx) != 1:
            raise ValueError(f"[ERROR] provenance 매칭 행 개수 이상: {row['fsd50k_fname']}")
        df.loc[ridx[0], "assigned_split"] = row["_new_split"]
        df.loc[ridx[0], "local_filename"] = row["_new_filename"]

    df.to_excel(provenance_path, index=False)

    remaining = os.listdir(staging)
    if remaining:
        raise RuntimeError(f"[ERROR] 스테이징 폴더에 파일이 남아있습니다: {remaining}")
    os.rmdir(staging)

    print(f"[완료] {len(new_rows)}개 파일 재분할 및 provenance 갱신: {provenance_path}")
    print("[다음 단계] generate_dataset_index.py를 재실행해 dataset_index CSV/PKL을 갱신하세요:")
    print(f"  python preprocessing/generate_dataset_index.py --mark_version {mark_version}")


def resplit_source_stratified(mark_version, seed, provenance_path,
                              val_per_class, test_per_class, dry_run=False):
    """[추가 2026-07-26] 원본(증강 제외)만 대상으로, 각 클래스 안에서 소스
    (fsd50k/aihub)를 층화해 val/test에 소스 비율대로 배정하고 나머지를 train으로.
    증강은 이 단계 이후 augment_density.py로 새 train 기준 재생성한다.

    배경: AI Hub 개951 편입 후 val/test가 AI Hub 편중(75%), train이 FSD50K 편중이
    되어 학습·평가 소스가 어긋났다(dev/eval 붕괴와 같은 도메인 시프트 위험). 각
    split에 두 소스를 같은 비율로 섞어 이 편중을 없앤다."""
    random.seed(seed)
    data_root = os.path.join(PROJECT_ROOT, "data")
    df = pd.read_excel(provenance_path)
    base_mask = (
        (df["mark_version"] == mark_version)
        & (df["removed_20260715"] == "active")
        & (df["source_type"] == "original")
    )
    base = df[base_mask]
    if base.empty:
        raise ValueError(f"[ERROR] active original 행 없음: mark_version='{mark_version}'")

    new_split_of = {}   # df index -> new split
    summary = {}        # (class, source, split) -> count
    for target_class, cls_group in base.groupby("target_class"):
        cls_total = len(cls_group)
        sources = sorted(cls_group["source"].unique())
        acc_val = acc_test = 0
        for i, src in enumerate(sources):
            src_idx = list(cls_group[cls_group["source"] == src].index)
            random.shuffle(src_idx)
            ssize = len(src_idx)
            if i < len(sources) - 1:      # 소스별 비례 배분, 마지막 소스가 잔차 흡수
                v = round(val_per_class * ssize / cls_total)
                t = round(test_per_class * ssize / cls_total)
            else:
                v = val_per_class - acc_val
                t = test_per_class - acc_test
            acc_val += v
            acc_test += t
            p = 0
            for _ in range(v):
                new_split_of[src_idx[p]] = "val"; p += 1
            for _ in range(t):
                new_split_of[src_idx[p]] = "test"; p += 1
            for j in range(p, ssize):
                new_split_of[src_idx[j]] = "train"
            summary[(target_class, src, "val")] = v
            summary[(target_class, src, "test")] = t
            summary[(target_class, src, "train")] = ssize - v - t

    print(f"[소스층화 재분할 계획] mark={mark_version}, val/test per class = {val_per_class}/{test_per_class}")
    for key in sorted(summary):
        cls, src, sp = key
        print(f"  {cls:10s} {src:7s} {sp:5s} : {summary[key]}")
    agg = {}
    for (cls, src, sp), c in summary.items():
        agg[(cls, sp)] = agg.get((cls, sp), 0) + c
    print("  --- split × class 합 (원본만) ---")
    for key in sorted(agg):
        print(f"  {key[0]:10s} {key[1]:5s} : {agg[key]}")

    if dry_run:
        print("[dry_run] 파일 이동·provenance 기록 없음.")
        return

    staging = os.path.join(PROJECT_ROOT, f"_resplit_staging_{mark_version}")
    os.makedirs(staging, exist_ok=True)
    moves = []
    for idx, new_split in new_split_of.items():
        row = df.loc[idx]
        old_path = os.path.join(data_root, row["assigned_split"], row["local_filename"])
        if not os.path.exists(old_path):
            raise FileNotFoundError(f"원본 파일 없음: {old_path}")
        staged = os.path.join(staging, row["local_filename"])
        shutil.move(old_path, staged)
        moves.append((idx, new_split, staged, row["target_class"]))

    counters = {}
    for idx, new_split, staged, cls in moves:
        counters[(cls, new_split)] = counters.get((cls, new_split), 0) + 1
        new_name = f"{cls}_{new_split}_{counters[(cls, new_split)]:03d}.wav"
        dst_dir = os.path.join(data_root, new_split)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.move(staged, os.path.join(dst_dir, new_name))
        df.loc[idx, "assigned_split"] = new_split
        df.loc[idx, "local_filename"] = new_name

    df.to_excel(provenance_path, index=False)
    if os.listdir(staging):
        raise RuntimeError(f"[ERROR] 스테이징 잔존: {os.listdir(staging)}")
    os.rmdir(staging)
    print(f"[완료] 원본 {len(moves)}개 소스층화 재분할 + provenance 갱신.")
    print("[다음] generate_dataset_index.py 재실행 → augment_density.py로 train 재증강.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FSD50K dev/eval 경계를 무시하고 클래스별 파일을 무작위로 train/val/test에 재배정합니다."
    )
    parser.add_argument("--mark_version", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--provenance_path",
        type=str,
        default=os.path.join(CODES_ROOT, "data_provenance.xlsx"),
        help="mark4/mark5 공용 데이터 출처 관리 엑셀 경로",
    )
    parser.add_argument(
        "--split_sizes",
        type=str,
        default="train:100,val:50,test:50",
        help="클래스당 split별 파일 수. 예: train:100,val:50,test:50",
    )
    parser.add_argument("--stratify_source", action="store_true",
                        help="원본(증강 제외)만 소스(fsd50k/aihub) 층화 재분할")
    parser.add_argument("--val_per_class", type=int, default=214)
    parser.add_argument("--test_per_class", type=int, default=214)
    parser.add_argument("--dry_run", action="store_true", help="파일 이동 없이 계획만 출력")
    args = parser.parse_args()

    if args.stratify_source:
        resplit_source_stratified(
            mark_version=args.mark_version,
            seed=args.seed,
            provenance_path=args.provenance_path,
            val_per_class=args.val_per_class,
            test_per_class=args.test_per_class,
            dry_run=args.dry_run,
        )
    else:
        parsed_sizes = []
        for item in args.split_sizes.split(","):
            name, count = item.split(":")
            parsed_sizes.append((name.strip(), int(count)))

        resplit(
            mark_version=args.mark_version,
            seed=args.seed,
            provenance_path=args.provenance_path,
            split_sizes=parsed_sizes,
        )
