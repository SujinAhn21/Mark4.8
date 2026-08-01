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

[추가 2026-08-01] --balanced 모드
위 기본 동작은 두 클래스 사이에 밀도 격차가 클 때(4.8: dog 0.52 vs others 0.67) 좁히는 용도다.
격차가 이미 거의 없을 때 같은 연산을 하면 반대로 없던 격차가 생긴다 — mark4.6(water_toilet)에서
원본 격차 -0.0096 이 증강 후 +0.1056 으로 벌어졌다. --balanced 는 클래스마다 증강본의 절반을
densify, 절반을 sparsify 로 만들어 평균 밀도는 유지하고 밀도 범위만 넓힌다.
같은 날, 증강 결과에도 check_waveform 게이트를 걸었다(그전엔 수집 스크립트에만 있었다).

사용 예:
  python preprocessing/augment_density.py --mark_version mark4.8
  python preprocessing/augment_density.py --mark_version mark4.8 --target_split train --n_aug_per_class 100 --seed 42
  python preprocessing/augment_density.py --mark_version mark4.6 --reset --balanced --target_per_class 1000
"""
import os
import sys
import glob
import time
import shutil
import hashlib
import argparse

import numpy as np
import soundfile as sf

# [추가 2026-08-01] 수집 단계(fsd50k_fetcher / aihub_slicer)와 같은 품질 기준을 쓰기 위해
# 판정 함수를 validate_dataset 과 공유한다. 2026-07-31 에 게이트를 넣을 때 이 스크립트만
# 빠져 있었고, 그 결과 mark4.6 에서 sparsify 결과의 peak 가 0.02 미만인(사실상 안 들리는)
# 증강본 5개가 그대로 저장됐다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from validate_dataset import check_waveform                      # noqa: E402

MAX_AUG_TRIES = 20          # 품질 미달이면 다른 원본으로 다시 만들어 보는 횟수


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
    parser.add_argument("--n_aug_per_class", type=int, default=100, help="클래스당 생성할 증강본 수(target_per_class 미지정 시)")
    parser.add_argument("--target_per_class", type=int, default=None,
                        help="지정 시 각 클래스를 이 수까지 채움(원본 부족분만큼 증강). 클래스별 원본 수가 달라도 목표 총량을 맞춘다.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--provenance_path", type=str, default=None,
                        help="data_provenance.xlsx 경로(기본: codes/data_provenance.xlsx)")
    parser.add_argument("--dry_run", action="store_true", help="파일/엑셀 안 쓰고 계획만 출력")
    parser.add_argument("--reset", action="store_true",
                        help="기존 증강본(파일 + provenance 행)을 먼저 지우고 새로 만든다. "
                             "재분할 뒤 재증강할 때 필요하다(안 지우면 이전 증강본이 새 train 구성과 "
                             "어긋난 채 남고 provenance 행도 중복된다)")
    parser.add_argument("--balanced", action="store_true",
                        help="클래스마다 증강본의 절반은 densify, 절반은 sparsify 로 만든다. "
                             "두 클래스의 밀도가 이미 비슷할 때 쓴다(한쪽만 빽빽하게/성기게 하면 "
                             "없던 격차가 생긴다). 기본 모드는 격차가 클 때 좁히는 용도다.")
    parser.add_argument("--no_quality_check", action="store_true",
                        help="증강 결과의 품질 검사를 끈다(권장하지 않음)")
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

    # ---- [추가 2026-07-31] --reset: 기존 증강본 제거 ----
    if args.reset:
        old_aug = sorted(glob.glob(os.path.join(split_dir, "*_aug_*.wav")))
        if not old_aug:
            print("[reset] 지울 기존 증강본이 없습니다.")
        elif args.dry_run:
            print(f"[reset/dry_run] 기존 증강본 {len(old_aug)}개와 그 provenance 행을 지울 예정")
        else:
            import pandas as pd
            names = {os.path.basename(p) for p in old_aug}
            if os.path.exists(prov_path):
                bak = prov_path + f".bak_before_reset_{time.strftime('%Y%m%d_%H%M%S')}"
                shutil.copy2(prov_path, bak)
                dfp = pd.read_excel(prov_path)
                if "source_type" in dfp.columns:
                    drop = (dfp["source_type"] == "augmented") & \
                           dfp["local_filename"].astype(str).isin(names)
                else:
                    drop = dfp["local_filename"].astype(str).isin(names)
                # [수정 2026-07-31] 지금 살아있는 증강본만 지운다. 파일명은 세대마다 재사용되므로
                # (예: others_train_aug_069) 이름만 보고 지우면 과거에 품질 문제로 제거된 행의
                # 이력까지 함께 사라진다. 실제로 이 조건이 없어서 446행이 지워졌는데 파일은
                # 428개였다(차액 18개가 옛 제거 이력).
                if "removed_20260715" in dfp.columns:
                    drop &= (dfp["removed_20260715"] == "active")
                # [추가 2026-08-01] 버전을 안 가리면 다른 mark 버전의 같은 이름 증강 행까지 지운다.
                # 재분할이 파일명을 클래스·split 별로 001 부터 다시 매기기 때문에
                # others_train_aug_001.wav 같은 이름은 모든 버전에 똑같이 생긴다.
                if "mark_version" in dfp.columns:
                    drop &= (dfp["mark_version"] == args.mark_version)
                dfp[~drop].to_excel(prov_path, index=False)
                print(f"[reset] provenance 증강 {int(drop.sum())}행 삭제 "
                      f"(백업 {os.path.basename(bak)})")
            for p in old_aug:
                os.remove(p)
            print(f"[reset] 기존 증강 wav {len(old_aug)}개 삭제 -> {split_dir}")

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

    # ---- 증강 수 결정 (target_per_class 지정 시 클래스별 부족분) ----
    if args.target_per_class is not None:
        n_sparse = args.target_per_class - len(by_class[sparse_cls])
        n_dense = args.target_per_class - len(by_class[dense_cls])
        if n_sparse < 0 or n_dense < 0:
            print(f"[ERROR] target_per_class={args.target_per_class}가 원본 수보다 작음 "
                  f"({sparse_cls}={len(by_class[sparse_cls])}, {dense_cls}={len(by_class[dense_cls])})")
            sys.exit(1)
    else:
        n_sparse = n_dense = args.n_aug_per_class
    print(f"  증강 수: {sparse_cls} +{n_sparse}, {dense_cls} +{n_dense}")

    # ---- 증강 생성 ----
    plan = []       # (out_path, waveform, method, source_base, active)
    rejected = []   # 품질 미달로 버린 시도 (클래스, 방식, 원본, 사유)

    def build(cls, n, method, src_list, target_active, start_idx=0):
        """cls 의 증강본을 method 방식으로 n 개 만든다.
        결과가 품질 기준에 미달하면 다른 원본으로 최대 MAX_AUG_TRIES 회까지 다시 만들어 본다.
        (sparsify 는 소리가 큰 청크를 지워버리면 남은 부분의 peak 가 기준 아래로 떨어질 수 있다.)"""
        made = 0
        for i in range(n):
            got = None
            for t in range(MAX_AUG_TRIES):
                srcpath, x, sr, _ = src_list[(i + t * 7) % len(src_list)]
                aug = (densify(x, sr, rng) if method == "densify"
                       else sparsify(x, sr, target_active, rng))
                bad = (None if args.no_quality_check
                       else check_waveform(np.asarray(aug, dtype=np.float32), sr))
                if bad is None:
                    got = (srcpath, sr, aug)
                    break
                rejected.append((cls, method, os.path.basename(srcpath), bad))
            if got is None:
                continue                     # 이 한 개는 채우지 못했다(아래에서 경고)
            srcpath, sr, aug = got
            out_base = f"{cls}_{args.target_split}_aug_{start_idx + made + 1:03d}.wav"
            plan.append((os.path.join(split_dir, out_base), aug, method,
                         os.path.basename(srcpath), active_ratio(aug, sr)))
            made += 1
        return made

    if args.balanced:
        # [추가 2026-08-01] 두 클래스의 밀도가 이미 비슷하면, 한쪽만 빽빽하게/한쪽만 성기게 만드는
        # 기본 모드가 없던 격차를 새로 만든다(mark4.6 실측: 원본 -0.0096 -> 증강 후 +0.1056).
        # 이 모드는 클래스마다 절반씩 densify/sparsify 해서 평균은 그대로 두고 밀도 범위만 넓힌다.
        print("  [balanced] 클래스마다 densify 절반 + sparsify 절반")
        for cls, n in ((sparse_cls, n_sparse), (dense_cls, n_dense)):
            items = by_class[cls]
            asc = sorted(items, key=lambda t: t[3])                  # 성긴 것부터 -> densify
            desc = sorted(items, key=lambda t: t[3], reverse=True)   # 빽빽한 것부터 -> sparsify
            n_dens = n // 2
            made_d = build(cls, n_dens, "densify", asc, means[cls])
            made_s = build(cls, n - n_dens, "sparsify", desc, means[cls], start_idx=made_d)
            print(f"    {cls}: densify {made_d}개 + sparsify {made_s}개")
    else:
        # 기본 모드: 성긴 클래스를 빽빽하게, 빽빽한 클래스를 성기게 해서 격차를 좁힌다.
        made_d = build(sparse_cls, n_sparse, "densify",
                       sorted(by_class[sparse_cls], key=lambda t: t[3]), sparse_target)
        made_s = build(dense_cls, n_dense, "sparsify",
                       sorted(by_class[dense_cls], key=lambda t: t[3], reverse=True), sparse_target)
        print(f"    {sparse_cls}: densify {made_d}개  /  {dense_cls}: sparsify {made_s}개")

    if rejected:
        print(f"[품질탈락] 증강 시도 {len(rejected)}건이 기준 미달이라 다른 원본으로 다시 만들었습니다:")
        for cls, method, src, why in rejected[:10]:
            print(f"  - {cls} {method} <- {src}: {why}")
    for cls, want in ((sparse_cls, n_sparse), (dense_cls, n_dense)):
        got = sum(1 for p in plan
                  if os.path.basename(p[0]).startswith(f"{cls}_{args.target_split}_aug_"))
        if got < want:
            print(f"[WARN] {cls}: 계획 {want}개 중 {got}개만 만들었습니다(품질 기준 통과 실패).")

    # ---- 증강 후 예상 분포 ----
    def all_active(cls):
        orig = [a for _, _, _, a in by_class[cls]]
        aug = [p[4] for p in plan if os.path.basename(p[0]).startswith(cls + f"_{args.target_split}_aug_")]
        return orig + aug
    da = all_active(sparse_cls); oa = all_active(dense_cls)
    print(f"[증강 후 예상] {sparse_cls} mean={np.mean(da):.4f}  {dense_cls} mean={np.mean(oa):.4f}  격차={np.mean(da)-np.mean(oa):+.4f}")
    print(f"  생성 예정 파일: {len(plan)}개 ({sparse_cls} {n_sparse} + {dense_cls} {n_dense})")

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
        # [추가 2026-08-01] 같은 파일명이 다른 mark 버전에도 있으므로 버전으로 좁힌다.
        # 안 좁히면 iloc[0] 이 다른 버전 행을 집어 라벨·출처를 엉뚱하게 물려받는다.
        src_row = df[df["local_filename"] == src_base]
        if "mark_version" in df.columns:
            src_row = src_row[src_row["mark_version"] == args.mark_version]
        orig_labels = src_row["original_labels"].iloc[0] if len(src_row) else ""
        src_source = src_row["source"].iloc[0] if (len(src_row) and "source" in df.columns) else ""
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
            "source": src_source,
            "removed_20260715": "active",
        })
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_excel(prov_path, index=False)
    print(f"[완료] data_provenance.xlsx에 증강본 {len(new_rows)}행 추가 (총 {len(df)}행) -> {prov_path}")
    print("[다음] generate_dataset_index.py 재실행 또는 run_all.py로 인덱스 갱신 필요.")


if __name__ == "__main__":
    main()
