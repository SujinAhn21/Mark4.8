"""
longwav_slicer.py — 라벨 없는 긴 wav 한 덩어리에서 3초 세그먼트를 뽑는 스크립트.

배경(2026-08-02): mark4.5(media_talking, TV·미디어 재생음)의 타겟 소리를 AI Hub 71296/585/71376,
FSD50K, TVSM, DEMAND 어디에서도 찾지 못했다. 공개 데이터셋은 TV 소리를 "제거할 배경잡음"으로만
다루기 때문에 분류 대상으로 라벨링된 것이 사실상 없다. 유일하게 쓸 만한 것이 VOiCES 의
distractors/rm4/tele — 방 안에서 TV 를 틀어놓고 마이크 여러 개로 55분씩 녹음한 파일이다.

aihub_slicer.py 를 쓸 수 없는 이유:
  - AI Hub 는 15초 클립 한 개당 annotation json 이 붙어 있어 "소리 구간"을 알 수 있다.
  - VOiCES 는 55분짜리 wav 한 개가 전부고 annotation 이 없다.
  - AI Hub 는 클립 하나가 곧 하나의 독립 녹음이라 클립 단위로 split 을 나누면 되지만,
    VOiCES 는 마이크 9개가 "같은 순간의 같은 TV 소리"를 위치만 바꿔 잡은 복제본이다.
    파일 단위로 split 을 나누면 train 의 t=100s 와 test 의 t=100s 가 같은 내용이 되어 누수가 난다.

그래서 이 스크립트는 split 을 시간축으로 나눈다.
  전체 길이를 --split_ratio 로 train / val / test 구간으로 자르고,
  각 구간을 3초 창으로 쭉 깔아 필요 개수만큼 균등 간격으로 고른다.
  창 하나에는 마이크를 라운드로빈으로 배정해, split 안에서 마이크가 골고루 섞이게 한다.
  → 어느 마이크를 쓰든 같은 시각은 같은 split 에만 들어가고(누수 차단),
    거리·잔향이 다른 마이크가 고루 들어간다.

  예) 3300초, 비율 0.5/0.25/0.25 일 때
      train 구간 [0, 1650)    → 3초 창 550개 (500개 필요, 예비 50)
      val   구간 [1650, 2475) → 창 275개 (215개 필요, 예비 60)
      test  구간 [2475, 3300) → val 과 동일

[변경 2026-08-02] 처음에는 "마이크 i = 시간조각 i" 로 마이크마다 다른 시간대를 고정 배정했는데,
녹음 레벨이 낮은 마이크(rm4/tele 의 mc18-mem-clo: peak 중앙값 0.0244 로 게이트 0.02 에 걸쳐
통과율 63%)가 맡은 시간대를 대체할 방법이 없어 930개 중 873개만 채워졌다. 지금처럼 창에
마이크를 배정하는 구조로 바꾸면, 창이 게이트에서 떨어졌을 때 같은 시각을 다른 마이크로 다시
읽어 볼 수 있다 — 내용은 같고 레벨만 다르니 대개 통과한다.

품질 게이트는 validate_dataset.check_waveform 을 그대로 쓴다(수집 기준 = 사후 검증 기준).
탈락하면 같은 칸에서 시간이 가장 가까운 예비 창으로 대체한다.

⚠ 이 버전은 resplit_dataset.py --stratify_source 를 돌리면 안 된다.
   그 스크립트는 클래스 안에서 파일을 무작위로 섞어 split 을 다시 배정하므로, 위에서 만든
   시간축 분리가 깨지고 인접 시각의 창이 train 과 test 로 흩어진다. mark4.5 는 타겟이 voices
   한 출처, others 가 aihub 한 출처뿐이라 소스 층화로 얻을 것도 없다.

provenance 컬럼은 aihub_slicer 와 같은 이름을 재사용한다(aihub_src_file / aihub_category /
aihub_volume). 컬럼을 새로 만들면 resplit·validate·augment 가 읽지 못하기 때문이다.
구분은 source 컬럼 값(voices)으로 한다.

사용 예:
  # 1) 타겟(media_talking) — VOiCES
  python preprocessing/longwav_slicer.py --mark_version mark4.5 \
      --wav_dir ~/voices/VOiCES_devkit/distant-16k/distractors/rm4/tele \
      --target_val 215 --target_test 215 --target_train 500 --dry_run

  # 2) others — AI Hub (aihub_slicer 에 타겟 0 을 주면 others 만 만든다)
  python preprocessing/aihub_slicer.py --mark_version mark4.5 \
      --target_val 0 --target_test 0 --target_train 0 \
      --others_val 215 --others_test 215 --others_train 500
"""
import os
import re
import sys
import time
import shutil
import hashlib
import argparse
from datetime import date

import numpy as np
import pandas as pd
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # preprocessing
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from validate_dataset import check_waveform                     # noqa: E402
# aihub_slicer 에서 target_class_from_config 등을 import 하지 않는 이유: 그 모듈은 최상위에서
# torch 를 import 하는데 /mnt/c 의 venv 에서는 그것만 76초가 걸린다(2026-07-31 실측).
# 짧은 함수 두 개는 여기에 그대로 둔다.

TARGET_SR = 16000
SEG_SEC = 3.0


def target_class_from_config(mark_version):
    """vild_config.py 에서 이 mark 버전의 타겟 클래스명을 읽는다(예: mark4.5 -> media_talking).

    소스를 텍스트로 읽어 정규식으로 뽑는다(torch 를 끌고 오지 않으려고).
    클래스명을 손으로 적지 않는 이유는 generate_dataset_index 의 키워드 맵이 같은 이름을 쓰기
    때문이다. 둘이 어긋나면 잘라낸 wav 가 인덱스에서 통째로 빠진다.
    """
    cfg_path = os.path.join(PROJECT_ROOT, "vild", "vild_config.py")
    src = open(cfg_path, encoding="utf-8").read()
    m = re.search(
        r'mark_version\s*==\s*["\']%s["\'][^\n]*\n\s*self\.classes\s*=\s*\[\s*["\']([^"\']+)["\']'
        % re.escape(mark_version), src)
    if not m:
        raise SystemExit(
            f"[ERROR] {cfg_path} 에서 '{mark_version}' 의 클래스 정의를 찾지 못했습니다.\n"
            f"        --target_class 로 직접 지정하십시오.")
    return m.group(1)


def _max_index(names, prefix):
    """names 중 prefix 로 시작하는 것들의 최대 번호. 증강본 _aug_ 제외."""
    mx = 0
    for p in names:
        if p.startswith(prefix) and p.endswith(".wav") and "_aug_" not in p:
            try:
                mx = max(mx, int(os.path.splitext(p)[0].rsplit("_", 1)[1]))
            except ValueError:
                pass
    return mx


def next_index(data_split_dir, prefix, prov_names=()):
    """이어붙일 다음 번호. 디스크 파일뿐 아니라 provenance 에 기록된 이름(제거된 행 포함)까지
    함께 봐서, 이전에 제거된 번호를 재사용하지 않는다."""
    disk = os.listdir(data_split_dir) if os.path.isdir(data_split_dir) else []
    return max(_max_index(disk, prefix), _max_index(prov_names, prefix)) + 1


def mic_id(filename):
    """파일명에서 마이크 식별자를 뽑는다(예: ...-mc04-lav-mid.wav -> mc04-lav-mid).
    규칙에 안 맞으면 확장자 뗀 파일명 전체를 쓴다."""
    m = re.search(r'(mc\d+[-\w]*)', os.path.splitext(filename)[0])
    return m.group(1) if m else os.path.splitext(filename)[0]


def scan_wavs(wav_dir, min_file_sec):
    """폴더의 wav 를 훑어 [(파일명, 길이초, 샘플레이트, 채널수)] 를 만든다.
    min_file_sec 미만(예: 다운로드가 잘린 파일)은 뺀다."""
    out, dropped = [], []
    for p in sorted(os.listdir(wav_dir)):
        if not p.lower().endswith(".wav"):
            continue
        try:
            info = sf.info(os.path.join(wav_dir, p))
        except Exception as e:
            dropped.append((p, str(e)))
            continue
        dur = info.frames / float(info.samplerate)
        if dur < min_file_sec:
            dropped.append((p, f"길이 {dur:.1f}s < {min_file_sec}s"))
            continue
        out.append((p, dur, info.samplerate, info.channels))
    return out, dropped


def read_window(path, start_sec, sr_file, channels):
    """파일에서 start_sec 부터 3초를 읽어 모노 float32 로 돌려준다(필요하면 16kHz 로 리샘플)."""
    x, sr = sf.read(path, start=int(round(start_sec * sr_file)),
                    frames=int(round(SEG_SEC * sr_file)), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != TARGET_SR:
        # 리샘플이 필요할 때만 torch 를 불러온다(import 만 76초).
        import torch
        import torchaudio.functional as AF
        x = AF.resample(torch.from_numpy(x).unsqueeze(0), sr, TARGET_SR).squeeze(0).numpy()
    return x.astype(np.float32)


def _pick_evenly(n_cands, need):
    """0..n_cands-1 에서 need 개를 균등 간격으로 고른다. (고른 것, 나머지)"""
    if need <= 0:
        return [], list(range(n_cands))
    if n_cands <= need:
        return list(range(n_cands)), []
    picked = sorted(set(np.linspace(0, n_cands - 1, need).round().astype(int)))
    # linspace 반올림이 겹치면 개수가 모자랄 수 있다 — 빈자리를 앞에서부터 채운다
    if len(picked) < need:
        for k in range(n_cands):
            if k not in picked:
                picked.append(k)
                if len(picked) == need:
                    break
        picked = sorted(picked)
    taken = set(picked)
    return picked, [k for k in range(n_cands) if k not in taken]


def plan_windows(files, split_counts, split_ratio):
    """시간축 split 분리 + 창마다 마이크 라운드로빈 배정.

    돌려주는 것: ({split: [창...]}, {split: [예비 창...]}, 부족목록)
    창 = {"start": 시작초, "mics": [(파일명, sr, ch), ...]}  ← 앞에서부터 시도할 순서
    같은 시각을 여러 마이크가 잡았으므로, 첫 마이크가 품질에서 떨어지면 다음 마이크로 넘어간다.
    """
    n_files = len(files)
    # 파일 길이가 다르면 가장 짧은 것에 맞춘다(모든 마이크가 커버하는 시간만 쓴다).
    dur = min(d for _, d, _, _ in files)
    b_tr = dur * split_ratio[0]
    b_va = dur * (split_ratio[0] + split_ratio[1])
    bounds = {"train": (0.0, b_tr), "val": (b_tr, b_va), "test": (b_va, dur)}

    plan, spare, shortage = {}, {}, []
    for sp in ("train", "val", "test"):
        need = int(split_counts[sp])
        s0, s1 = bounds[sp]
        grid = [s0 + k * SEG_SEC for k in range(int((s1 - s0) // SEG_SEC))]
        if need > len(grid):
            shortage.append((sp, need, len(grid)))
        picked_idx, spare_idx = _pick_evenly(len(grid), need)

        def make(rank, k):
            # rank 번째 창의 1순위 마이크는 rank % n_files. 실패하면 그다음 마이크로.
            order = [(rank + j) % n_files for j in range(n_files)]
            return {"start": grid[k],
                    "mics": [(files[i][0], files[i][2], files[i][3]) for i in order]}

        plan[sp] = [make(r, k) for r, k in enumerate(picked_idx)]
        spare[sp] = [make(r, k) for r, k in enumerate(spare_idx)]
    return plan, spare, shortage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mark_version", type=str, required=True)
    ap.add_argument("--wav_dir", type=str, required=True,
                    help="긴 wav 들이 들어 있는 폴더(파일 하나 = 마이크 하나)")
    ap.add_argument("--provenance_path", type=str,
                    default=os.path.join(PROJECT_ROOT, "..", "data_provenance.xlsx"))
    ap.add_argument("--target_class", type=str, default=None,
                    help="타겟 클래스명. 생략하면 vild_config.py 에서 mark_version 으로 읽는다.")
    ap.add_argument("--source_name", type=str, default="voices",
                    help="provenance 의 source 컬럼에 넣을 값(resplit 소스 층화가 이 값을 본다)")
    ap.add_argument("--src_category", type=str, default=None,
                    help="provenance 의 aihub_category 에 넣을 값. 생략하면 wav_dir 마지막 두 단계.")
    ap.add_argument("--label_text", type=str, default="television",
                    help="provenance 의 original_labels 에 넣을 값")
    ap.add_argument("--target_val", type=int, default=215)
    ap.add_argument("--target_test", type=int, default=215)
    ap.add_argument("--target_train", type=int, default=500)
    ap.add_argument("--split_ratio", type=str, default="0.5,0.25,0.25",
                    help="train,val,test 시간 비율. 개수 비율보다 여유 있게 잡아야 품질 탈락을 흡수한다.")
    ap.add_argument("--min_file_sec", type=float, default=600.0,
                    help="이보다 짧은 wav 는 제외(다운로드가 잘린 파일 방지)")
    # 창 선택은 균등 간격, 마이크 배정은 라운드로빈이라 무작위가 없다(--seed 없음).
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--no_quality_check", action="store_true",
                    help="잘라낸 3초의 품질 검사를 끈다(권장하지 않음)")
    args = ap.parse_args()

    prov_path = os.path.abspath(args.provenance_path)
    wav_dir = os.path.abspath(os.path.expanduser(args.wav_dir))
    target_class = args.target_class or target_class_from_config(args.mark_version)
    ratio = [float(v) for v in args.split_ratio.split(",")]
    if len(ratio) != 3 or abs(sum(ratio) - 1.0) > 1e-6:
        raise SystemExit(f"[ERROR] --split_ratio 는 합이 1 인 세 값이어야 합니다: {args.split_ratio}")
    src_category = args.src_category or "/".join(wav_dir.rstrip("/").split(os.sep)[-2:])
    print(f"[INFO] mark_version={args.mark_version}  타겟 클래스={target_class}  "
          f"source={args.source_name}  category={src_category}")

    # 1) provenance 대조: 고아 파일 점검 + 이 버전이 이미 쓴 창 파악
    prov = pd.read_excel(prov_path)
    mine = prov[prov["mark_version"] == args.mark_version] if "mark_version" in prov.columns else prov
    active = mine[mine["removed_20260715"] == "active"] if "removed_20260715" in mine.columns else mine
    known_files = set(active["local_filename"].astype(str))
    for sp in ("train", "val", "test"):
        sp_dir = os.path.join(PROJECT_ROOT, "data", sp)
        orphans = [p for p in (os.listdir(sp_dir) if os.path.isdir(sp_dir) else [])
                   if p.endswith(".wav") and p not in known_files]
        if orphans:
            print(f"[ERROR] provenance 에 없는 wav 가 data/{sp} 에 {len(orphans)}개 있습니다"
                  f"(이전 실행이 중간에 끊긴 흔적):")
            for p in orphans[:10]:
                print("  -", p)
            print("이 파일들을 정리(삭제 또는 provenance 반영)한 뒤 다시 실행하십시오.")
            sys.exit(1)
    # 이미 쓴 창은 (원본 파일명, 시작초) 로 식별한다. 같은 파일의 다른 시각은 다른 데이터다.
    used_win = set()
    if {"aihub_src_file", "seg_start_sec"} <= set(mine.columns):
        got = mine.loc[mine["aihub_src_file"].notna(), ["aihub_src_file", "seg_start_sec"]]
        used_win = {(str(a), round(float(b), 3))
                    for a, b in got.itertuples(index=False, name=None)
                    if pd.notna(b)}
    print(f"[INFO] provenance 전체 {len(prov)}행 중 {args.mark_version} {len(mine)}행"
          f"(active {len(active)}), 이 버전이 이미 쓴 창 {len(used_win)}개")

    # 2) 파일 스캔
    files, dropped = scan_wavs(wav_dir, args.min_file_sec)
    if dropped:
        print(f"[제외] {len(dropped)}개 파일:")
        for p, why in dropped[:10]:
            print(f"  - {p}: {why}")
    if not files:
        raise SystemExit(f"[ERROR] {wav_dir} 에 {args.min_file_sec}s 이상인 wav 가 없습니다.")
    total_sec = sum(d for _, d, _, _ in files)
    print(f"[INFO] 대상 wav {len(files)}개, 합계 {total_sec/60:.1f}분 "
          f"(파일당 {files[0][1]/60:.1f}분, {files[0][2]}Hz, {files[0][3]}ch)")

    # 3) 계획: 시간축 split → 창 나열 → 균등 선택 → 마이크 라운드로빈
    split_counts = {"train": args.target_train, "val": args.target_val, "test": args.target_test}
    plan, spare, shortage = plan_windows(files, split_counts, ratio)
    print(f"[계획] 시간 비율 train/val/test = {ratio[0]:.2f}/{ratio[1]:.2f}/{ratio[2]:.2f}"
          f"  (같은 시각은 한 split 에만 들어간다)")
    for sp in ("train", "val", "test"):
        print(f"  {sp:5s}: {len(plan[sp])}개 채택, 예비 창 {len(spare[sp])}개 "
              f"(창마다 마이크 {len(files)}개를 순서대로 시도)")
    if shortage:
        print("[ERROR] split 구간이 짧아 필요 개수를 못 채웁니다:")
        for sp, need, have in shortage:
            print(f"  - {sp}: 필요 {need}, 창 {have}")
        print("        --split_ratio 를 조정하거나 개수를 줄이십시오.")
        sys.exit(1)
    if args.dry_run:
        print("[DRY RUN] 추출은 하지 않고 종료합니다.")
        return

    # 4) 추출 + 저장 + provenance 행 추가
    backup = prov_path + f".bak_before_longwav_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(prov_path, backup)
    print(f"[INFO] provenance 백업: {os.path.basename(backup)}")
    if "source" not in prov.columns:
        prov["source"] = "fsd50k"
    # 번호는 일부러 버전으로 거르지 않고 전 버전 통틀어 이어붙인다(파일명 충돌 방지).
    prov_names = set(prov["local_filename"].astype(str))

    counters, new_rows, rejected, unfilled = {}, [], [], []
    mic_used = {}
    t0 = time.time()
    todo = sum(len(plan[sp]) for sp in ("val", "test", "train"))
    done = 0

    def try_window(win):
        """창 하나를 마이크 순서대로 시도해 (파일명, 시작초, 파형) 을 돌려준다. 전부 실패면 None."""
        for fname, sr_file, ch in win["mics"]:
            start = win["start"]
            if (fname, round(start, 3)) in used_win:
                rejected.append((fname, round(start, 1), "이미쓴창"))
                continue
            try:
                seg = read_window(os.path.join(wav_dir, fname), start, sr_file, ch)
            except Exception as e:
                rejected.append((fname, round(start, 1), f"읽기실패 {e}"))
                continue
            bad = (None if args.no_quality_check
                   else check_waveform(seg, TARGET_SR, min_duration=SEG_SEC))
            if bad is None:
                return fname, start, seg
            rejected.append((fname, round(start, 1), bad))
        return None

    for sp in ("val", "test", "train"):
        data_dir = os.path.join(PROJECT_ROOT, "data", sp)
        os.makedirs(data_dir, exist_ok=True)
        key = (sp, target_class)
        counters[key] = next_index(data_dir, f"{target_class}_{sp}_", prov_names)
        pool = list(spare[sp])          # 이 split 의 예비 창
        for win in plan[sp]:
            got = try_window(win)
            while got is None and pool:
                # 마이크를 다 써도 안 되면, 시간이 가장 가까운 예비 창으로 옮긴다
                j = int(np.argmin([abs(w["start"] - win["start"]) for w in pool]))
                got = try_window(pool.pop(j))
            done += 1
            if got is None:
                unfilled.append((sp, round(win["start"], 1)))
                continue
            fname, start, seg = got
            mic = mic_id(fname)
            mic_used[mic] = mic_used.get(mic, 0) + 1
            local_name = f"{target_class}_{sp}_{counters[key]:03d}.wav"
            out_path = os.path.join(data_dir, local_name)
            sf.write(out_path, seg, TARGET_SR, subtype="FLOAT")
            counters[key] += 1
            used_win.add((fname, round(start, 3)))
            new_rows.append({
                "local_filename": local_name, "original_labels": args.label_text,
                "target_class": target_class, "assigned_split": sp,
                "mark_version": args.mark_version,
                "sha256": hashlib.sha256(open(out_path, "rb").read()).hexdigest(),
                "size_bytes": os.path.getsize(out_path),
                "download_date": date.today().isoformat(), "source_type": "original",
                "removed_20260715": "active",
                "source": args.source_name, "aihub_src_file": fname,
                "aihub_category": src_category, "aihub_volume": mic,
                "seg_start_sec": round(start, 3),
                "seg_end_sec": round(start + SEG_SEC, 3),
            })
            if done % 200 == 0 or done == todo:
                merged = pd.concat([prov, pd.DataFrame(new_rows)], ignore_index=True)
                merged.to_excel(prov_path, index=False)
                print(f"[진행] {done}/{todo} — {(time.time()-t0)/60:.1f}분 경과, "
                      f"provenance 중간 저장")

    if new_rows:
        merged = pd.concat([prov, pd.DataFrame(new_rows)], ignore_index=True)
        merged.to_excel(prov_path, index=False)

    print(f"\n[완료] 신규 저장 {len(new_rows)}개")
    made = {}
    for r in new_rows:
        made[r["assigned_split"]] = made.get(r["assigned_split"], 0) + 1
    print("[실제] " + ", ".join(f"{sp}/{target_class} +{n}" for sp, n in sorted(made.items())))
    print("[마이크별 채택 수] " + ", ".join(f"{m} {n}" for m, n in sorted(mic_used.items())))
    if rejected:
        print(f"[품질탈락] {len(rejected)}회(다음 마이크나 가까운 예비 창으로 대체):")
        by_mic = {}
        for f, s, why in rejected:
            by_mic[mic_id(f)] = by_mic.get(mic_id(f), 0) + 1
        print("  마이크별: " + ", ".join(f"{m} {n}" for m, n in sorted(by_mic.items())))
        for f, s, why in rejected[:5]:
            print(f"  - {f} t={s}s: {why}")
    if unfilled:
        print(f"[WARN] 마이크와 예비 창을 다 써도 못 채운 자리 {len(unfilled)}개:")
        for sp, s in unfilled[:10]:
            print(f"  - {sp} t={s}s")
    print(f"provenance: {prov_path} ({len(prov) + len(new_rows)}행)")
    print("다음 단계: others 는 aihub_slicer.py(타겟 0개)로, 그 뒤 generate_dataset_index.py.")
    print("⚠ resplit_dataset.py 는 돌리지 마십시오 — 시간축 split 분리가 깨집니다.")


if __name__ == "__main__":
    main()
