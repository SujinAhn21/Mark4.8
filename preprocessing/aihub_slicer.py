"""
aihub_slicer.py — AI Hub 생활환경소음(dataSetSn=71296) 로컬 원본에서 3초 세그먼트를 추출하는 스크립트.

배경(2026-07-15): FSD50K 개 소리(Bark+Dog)가 전량 소진되어 val/test 확장이 막힘.
AI Hub 71296(주거공간 실녹음, 전 클립 15초/48kHz/16bit/모노, 소리 구간 시각 주석 포함)을
새 출처로 도입한다. 원본 zip 다운로드·해제는 별도(세션 로그 260715_(2) 참조)로 끝났고,
이 스크립트는 해제된 로컬 폴더에서 라벨 annotation을 읽어 3초 세그먼트만 잘라낸다.

동작:
1. data_provenance.xlsx 의 active 행과 data/{train,val,test} 실파일을 대조(고아 파일 시 중단),
   aihub_src_file 로 이미 쓴 원본 클립은 건너뛴다(중복 방지).
2. 원본 클립 풀을 만든다.
   - 타겟 클래스: --target_keyword 로 고른 AI Hub 카테고리(TS+VS)를 섞어 val/test/train 에 배타 배정.
     예) mark4.8 은 '강아지'(VS 106 + TS 845 = 951클립), mark4.6 은 '화장실물내리는소리'(VS 105 + TS 839).
   - others: 타겟 카테고리를 뺀 VS 카테고리에서 카테고리 균등 배분으로 val/test/train 배정.
   같은 원본 클립은 한 split 에만 들어간다(클립당 세그먼트 1개).
3. 라벨 json 의 annotation 중 가장 긴 구간의 중앙에 3초 창을 맞춰(클립 경계로 클램프) 자르고,
   48kHz -> 16kHz 리샘플 + 모노 + float32 로 data/{split}/{class}_{split}_{NNN}.wav 저장
   (기존 FSD50K 파일 번호에 이어붙임). 창의 RMS 가 너무 낮으면 다음 annotation 으로 폴백.
4. data_provenance.xlsx 에 행 추가(백업 생성, 200개마다 중간 저장). 신규 컬럼:
   source(aihub; 기존 행은 fsd50k 로 채움), aihub_src_file, aihub_category, aihub_volume(TS/VS),
   seg_start_sec, seg_end_sec.

주의: AI Hub 데이터는 재배포 금지 — 생성된 wav 를 wav 커밋 repo(Mark4.8)에 푸시하지 않는다.

[변경 2026-08-01] mark4.1~4.7 도 같은 코드로 수집하기 위해 타겟 클래스를 인자로 뺐다.
이전에는 DOG_KEYWORD = "강아지" 와 --dog_* 인자가 소스에 박혀 있어 4.8 전용이었다.
타겟 클래스명은 --target_class 로 직접 줄 수도 있고, 안 주면 vild_config.py 에서 읽는다
(generate_dataset_index 의 키워드 맵과 어긋나지 않게 하려는 것이다).

사용 예:
  python preprocessing/aihub_slicer.py --mark_version mark4.8 --target_keyword 강아지 --dry_run
  python preprocessing/aihub_slicer.py --mark_version mark4.6 --target_keyword 화장실물내리는소리 \
      --target_val 215 --target_test 215 --target_train 514 \
      --others_val 215 --others_test 215 --others_train 514 --dry_run
"""
import os
import re
import sys
import json
import time
import shutil
import hashlib
import argparse
from datetime import date

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as AF

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # preprocessing
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# 수집 기준과 사후 검증 기준이 어긋나지 않도록 판정 함수를 validate_dataset 과 공유한다.
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from validate_dataset import check_waveform                     # noqa: E402

TARGET_SR = 16000
SEG_SEC = 3.0
MIN_SEG_RMS = 1e-3          # 이보다 낮으면 사실상 무음 창으로 보고 다른 annotation 시도


def target_class_from_config(mark_version):
    """vild_config.py 에서 이 mark 버전의 타겟 클래스명을 읽는다(예: mark4.6 -> water_toilet).

    vild_config 를 import 하지 않고 소스를 텍스트로 읽어 정규식으로 뽑는다. import 하면 torch 까지
    딸려오는데 /mnt/c 의 venv 에서는 그것만 76초가 걸린다(2026-07-31 실측).
    클래스명을 손으로 적지 않고 config 에서 읽는 이유는, generate_dataset_index 의 키워드 맵이
    같은 이름을 쓰기 때문이다. 둘이 어긋나면 잘라낸 wav 가 인덱스에서 통째로 빠진다.
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
    """names(파일명 iterable) 중 prefix 로 시작하는 것들의 최대 번호. 증강본 _aug_ 제외."""
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
    함께 봐서, 이전에 제거된 번호(예: 클리핑으로 뺀 050)를 재사용하지 않는다."""
    disk = os.listdir(data_split_dir) if os.path.isdir(data_split_dir) else []
    return max(_max_index(disk, prefix), _max_index(prov_names, prefix)) + 1


def scan_aihub(root):
    """해제 폴더를 스캔해 {카테고리명: {"volume": TS|VS, "dir": 경로, "wavs": [파일명...]}} 목록을 만든다.
    카테고리명은 폴더명에서 TS_/VS_ 접두어를 뗀 것(라벨 폴더는 _label 로 구분)."""
    cats = []
    for d in sorted(os.listdir(root)):
        full = os.path.join(root, d)
        if not os.path.isdir(full) or d.endswith("_label"):
            continue
        vol = d[:2]
        if vol not in ("TS", "VS"):
            continue
        name = d[3:]
        label_dir = os.path.join(root, f"{'TL' if vol == 'TS' else 'VL'}_{name}_label")
        wavs = sorted(f for f in os.listdir(full) if f.endswith(".wav"))
        if not wavs:
            continue
        if not os.path.isdir(label_dir):
            print(f"[WARN] 라벨 폴더 없음(annotation 없이 중앙 창 사용): {d}")
            label_dir = None
        cats.append({"volume": vol, "category": name, "dir": full,
                     "label_dir": label_dir, "wavs": wavs})
    return cats


def pick_window(wav_path, label_path, rng):
    """클립에서 3초 창(시작초, 끝초)과 라벨텍스트를 고른다.
    가장 긴 annotation 중앙 우선, 창 RMS 미달 시 다음 annotation, 전부 미달이면 클립 중앙."""
    info = sf.info(wav_path)
    dur = info.frames / info.samplerate
    anns = []
    if label_path and os.path.exists(label_path):
        with open(label_path, encoding="utf-8") as f:
            anns = json.load(f).get("annotation", [])
        anns = sorted(anns, key=lambda a: a["endTime"] - a["startTime"], reverse=True)

    data, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    x = data.mean(axis=1)

    def window_at(center):
        start = min(max(center - SEG_SEC / 2, 0.0), max(dur - SEG_SEC, 0.0))
        end = min(start + SEG_SEC, dur)
        return start, end

    candidates = [window_at((a["startTime"] + a["endTime"]) / 2) for a in anns]
    candidates.append(window_at(dur / 2))
    label_texts = [a.get("labelText", "") for a in anns] + [""]

    for (start, end), text in zip(candidates, label_texts):
        seg = x[int(start * sr): int(end * sr)]
        if len(seg) and float(np.sqrt(np.mean(seg ** 2))) >= MIN_SEG_RMS:
            return x, sr, dur, start, end, text
    # 전 후보가 무음에 가까움 — 첫 후보를 그대로 쓰되 호출부에서 경고
    start, end = candidates[0]
    return x, sr, dur, start, end, label_texts[0]


def slice_resample(x, sr, start, end):
    seg = x[int(start * sr): int(end * sr)]
    if sr != TARGET_SR:
        t = torch.from_numpy(seg).unsqueeze(0)
        seg = AF.resample(t, sr, TARGET_SR).squeeze(0).numpy()
    return seg.astype(np.float32)


def distribute_evenly(total, n_cats, rng):
    """total 개를 n_cats 카테고리에 균등 배분(나머지는 무작위 카테고리에 +1)."""
    quota = np.full(n_cats, total // n_cats, dtype=int)
    extra = rng.choice(n_cats, size=total % n_cats, replace=False)
    quota[extra] += 1
    return quota


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mark_version", type=str, required=True)
    ap.add_argument("--aihub_root", type=str,
                    default=os.path.expanduser("~/workspace/aihub_raw/71296/extracted"))
    ap.add_argument("--provenance_path", type=str,
                    default=os.path.join(PROJECT_ROOT, "..", "data_provenance.xlsx"))
    ap.add_argument("--target_class", type=str, default=None,
                    help="타겟 클래스명. 생략하면 vild_config.py 에서 mark_version 으로 읽는다.")
    ap.add_argument("--target_keyword", type=str, required=True,
                    help="AI Hub 카테고리 폴더명에 들어 있는 문자열(부분일치). 쉼표로 여러 개 가능. "
                         "예: 강아지 / 화장실물내리는소리 / 어른발걸음소리,아이들발걸음소리,망치질소리")
    # 2026-07-15 확정 계획(mark4.8): val 타겟+150/others+151, test +151/+152, train +400/+400
    # --dog_* 는 4.8 때 쓰던 이름으로, 기존 명령이 그대로 돌아가도록 별칭으로 남겨둔다.
    ap.add_argument("--target_val", "--dog_val", dest="target_val", type=int, default=150)
    ap.add_argument("--target_test", "--dog_test", dest="target_test", type=int, default=151)
    ap.add_argument("--target_train", "--dog_train", dest="target_train", type=int, default=400)
    ap.add_argument("--others_val", type=int, default=151)
    ap.add_argument("--others_test", type=int, default=152)
    ap.add_argument("--others_train", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--no_quality_check", action="store_true",
                    help="잘라낸 3초의 품질 검사를 끄고 예전처럼 그대로 저장(권장하지 않음)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    prov_path = os.path.abspath(args.provenance_path)
    target_class = args.target_class or target_class_from_config(args.mark_version)
    keywords = [k.strip() for k in args.target_keyword.split(",") if k.strip()]
    if not keywords:
        raise SystemExit("[ERROR] --target_keyword 가 비어 있습니다.")
    print(f"[INFO] mark_version={args.mark_version}  타겟 클래스={target_class}  "
          f"카테고리 키워드={keywords}")

    # 1) provenance 대조: 고아 파일 점검 + 이미 쓴 aihub 원본 클립 파악
    prov = pd.read_excel(prov_path)
    # [변경 2026-08-01] provenance 는 mark4.x 전 버전이 한 파일을 공유한다. 고아 검사와 '이미 쓴 원본'
    # 판정은 이 버전의 행만 봐야 한다. 버전을 안 거르면 4.6 을 수집할 때 4.8 이 쓴 others 클립까지
    # 사용 불가로 잡혀서, 버전이 늘수록 others 풀이 말라붙는다(7개 버전이면 가용 4,714개를 넘어 고갈).
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
    used_src = set()
    if "aihub_src_file" in mine.columns:
        used_src = set(mine.loc[mine["aihub_src_file"].notna(), "aihub_src_file"].astype(str))
    print(f"[INFO] provenance 전체 {len(prov)}행 중 {args.mark_version} {len(mine)}행"
          f"(active {len(active)}), 이 버전이 이미 쓴 aihub 원본 {len(used_src)}개")

    # 2) 원본 스캔 및 풀 구성
    cats = scan_aihub(args.aihub_root)
    target_pool = []    # (volume, category, dir, label_dir, wav)
    others_cats = []    # 타겟 카테고리를 뺀 VS 카테고리
    matched_cats = []
    for c in cats:
        wavs = [w for w in c["wavs"] if w not in used_src]
        if any(k in c["category"] for k in keywords):
            target_pool += [(c["volume"], c["category"], c["dir"], c["label_dir"], w) for w in wavs]
            matched_cats.append(f"{c['volume']}:{c['category']}({len(wavs)})")
        elif c["volume"] == "VS":
            others_cats.append((c, wavs))
    if not matched_cats:
        print(f"[ERROR] --target_keyword {keywords} 에 맞는 카테고리가 {args.aihub_root} 에 없습니다.")
        print("        해제된 카테고리 폴더 목록:")
        for c in cats:
            print(f"          {c['volume']}_{c['category']}")
        sys.exit(1)
    print(f"[INFO] 타겟({target_class}) 카테고리 {len(matched_cats)}개: {', '.join(matched_cats)}")
    print(f"[INFO] 타겟 풀 {len(target_pool)}클립, others 카테고리 {len(others_cats)}개 "
          f"(클립 {sum(len(w) for _, w in others_cats)}개)")

    # 타겟: 풀 전체를 섞어 val/test/train 배타 배정
    need_target = args.target_val + args.target_test + args.target_train
    if len(target_pool) < need_target:
        print(f"[ERROR] {target_class} 클립 부족: 필요 {need_target}, 가용 {len(target_pool)}")
        sys.exit(1)
    target_pool.sort(key=lambda t: (t[0], t[1], t[4]))
    rng.shuffle(target_pool)
    jobs = []           # (split, class, volume, category, dir, label_dir, wav)
    # [추가 2026-07-31] 품질 미달 클립을 대체할 예비 풀. 키는 (타겟클래스, None) / ("others", 카테고리).
    # 이전에는 전 창이 무음에 가까워도 경고만 찍고 그대로 저장해서, 2026-07-31 수집분 117개 중
    # 3개가 rms 0.0006~0.001(사실상 안 들림)로 학습 데이터에 들어갔다.
    spare = {}
    pos = 0
    for sp, n in (("val", args.target_val), ("test", args.target_test),
                  ("train", args.target_train)):
        jobs += [(sp, target_class) + t for t in target_pool[pos:pos + n]]
        pos += n
    spare[(target_class, None)] = list(target_pool[pos:])

    # others: 카테고리 균등 배분(한 카테고리 안에서 셔플 후 val->test->train 순서로 잘라 배타 보장)
    n_cats = len(others_cats)
    quotas = {sp: distribute_evenly(n, n_cats, rng)
              for sp, n in (("val", args.others_val), ("test", args.others_test),
                            ("train", args.others_train))}
    for i, (c, wavs) in enumerate(others_cats):
        need = sum(int(quotas[sp][i]) for sp in ("val", "test", "train"))
        if len(wavs) < need:
            print(f"[ERROR] others '{c['category']}' 클립 부족: 필요 {need}, 가용 {len(wavs)}")
            sys.exit(1)
        wavs = sorted(wavs)
        rng.shuffle(wavs)
        p = 0
        for sp in ("val", "test", "train"):
            q = int(quotas[sp][i])
            jobs += [(sp, "others", c["volume"], c["category"], c["dir"], c["label_dir"], w)
                     for w in wavs[p:p + q]]
            p += q
        spare[("others", c["category"])] = [
            (c["volume"], c["category"], c["dir"], c["label_dir"], w) for w in wavs[p:]]

    plan = {}
    for j in jobs:
        plan[(j[0], j[1])] = plan.get((j[0], j[1]), 0) + 1
    print("[계획] " + ", ".join(f"{sp}/{cls} +{n}" for (sp, cls), n in sorted(plan.items())))
    if args.dry_run:
        print("[DRY RUN] 추출은 하지 않고 종료합니다.")
        return

    # 3) 추출 + 저장 + provenance 행 추가
    backup = prov_path + f".bak_before_aihub_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(prov_path, backup)
    print(f"[INFO] provenance 백업: {os.path.basename(backup)}")
    if "source" not in prov.columns:
        prov["source"] = "fsd50k"      # 기존 행 전부 FSD50K 유래(증강본 포함)

    # provenance 에 이미 기록된 모든 이름(제거된 행 포함) — 번호 재사용 방지에 사용.
    # [주의 2026-08-01] 여기는 일부러 버전으로 거르지 않는다. validate_dataset 의 --update_provenance /
    # --quarantine 이 local_filename 만 보고 행을 찾기 때문에, 버전이 달라도 파일명이 겹치면 다른
    # 버전 행에 측정치가 덮이거나 엉뚱한 행이 제거로 표시된다(others_train_001.wav 는 모든 버전에
    # 생긴다). 그래서 번호를 전 버전 통틀어 이어붙여 파일명이 겹치지 않게 한다.
    prov_names = set(prov["local_filename"].astype(str))

    counters, new_rows, warned, skipped, rejected = {}, [], [], [], []
    t0 = time.time()
    for i, (sp, cls, vol, cat, cdir, ldir, wav) in enumerate(jobs, 1):
        data_dir = os.path.join(PROJECT_ROOT, "data", sp)
        os.makedirs(data_dir, exist_ok=True)
        key = (sp, cls)
        if key not in counters:
            counters[key] = next_index(data_dir, f"{cls}_{sp}_", prov_names)
        # [변경 2026-07-31] 잘라낸 3초가 품질 기준에 미달하면 저장하지 않고, 같은 카테고리의
        # 예비 클립으로 대체한다(예비가 떨어지면 그때 포기). 기준은 validate_dataset 과 공유한다.
        spare_key = (target_class, None) if cls == target_class else ("others", cat)
        cur = (vol, cat, cdir, ldir, wav)
        seg = local_name = out_path = None
        tries = 0
        while True:
            tries += 1
            _vol, _cat, _cdir, _ldir, _wav = cur
            try:
                label_path = (os.path.join(_ldir, os.path.splitext(_wav)[0] + ".json")
                              if _ldir else None)
                x, sr, dur, start, end, text = pick_window(
                    os.path.join(_cdir, _wav), label_path, rng)
                cand = slice_resample(x, sr, start, end)
            except Exception as e:
                # 원본 클립 하나가 깨져도(예: Format not recognised) 전체를 죽이지 않는다.
                skipped.append((sp, cls, _cat, _wav, str(e)))
                print(f"[SKIP] {sp}/{cls} {_cat}/{_wav}: {e}")
                cand = None
            if cand is not None:
                bad = (check_waveform(cand, TARGET_SR, min_duration=SEG_SEC)
                       if not args.no_quality_check else None)
                if bad is None:
                    seg, wav, cat, vol = cand, _wav, _cat, _vol
                    break
                rejected.append((_wav, _cat, bad))
            if not spare.get(spare_key):
                break                       # 대체할 예비가 없다
            cur = spare[spare_key].pop()
        if seg is None:
            continue                        # 이 job 은 채우지 못했다(마지막에 경고)
        if float(np.sqrt(np.mean(seg ** 2))) < MIN_SEG_RMS:
            warned.append((wav, cat))
        local_name = f"{cls}_{sp}_{counters[key]:03d}.wav"
        out_path = os.path.join(data_dir, local_name)
        sf.write(out_path, seg, TARGET_SR, subtype="FLOAT")
        counters[key] += 1
        new_rows.append({
            "local_filename": local_name, "original_labels": text or cat.split("_")[-1],
            "target_class": cls, "assigned_split": sp, "mark_version": args.mark_version,
            "sha256": hashlib.sha256(open(out_path, "rb").read()).hexdigest(),
            "size_bytes": os.path.getsize(out_path),
            "download_date": date.today().isoformat(), "source_type": "original",
            "removed_20260715": "active",
            "source": "aihub", "aihub_src_file": wav, "aihub_category": cat,
            "aihub_volume": vol, "seg_start_sec": round(start, 3), "seg_end_sec": round(end, 3),
        })
        if i % 200 == 0 or i == len(jobs):
            merged = pd.concat([prov, pd.DataFrame(new_rows)], ignore_index=True)
            merged.to_excel(prov_path, index=False)
            print(f"[진행] {i}/{len(jobs)} — {(time.time()-t0)/60:.1f}분 경과, provenance 중간 저장")

    print(f"\n[완료] 신규 저장 {len(new_rows)}개")
    # 실제 저장된 split/class 수량 대조
    made = {}
    for r in new_rows:
        made[(r["assigned_split"], r["target_class"])] = made.get(
            (r["assigned_split"], r["target_class"]), 0) + 1
    print("[실제] " + ", ".join(f"{sp}/{cls} +{n}" for (sp, cls), n in sorted(made.items())))
    if skipped:
        print(f"[WARN] 원본 손상 등으로 건너뛴 클립 {len(skipped)}개(계획보다 그만큼 부족):")
        for sp, cls, cat, w, e in skipped[:20]:
            print(f"  - {sp}/{cls} {cat}/{w}: {e}")
    if rejected:
        print(f"[품질탈락] {len(rejected)}개(저장하지 않고 예비 클립으로 대체):")
        for w, c, r in rejected[:10]:
            print(f"  - {w} ({c}): {r}")
    for (sp, cls), want in sorted(plan.items()):
        got = made.get((sp, cls), 0)
        if got < want:
            print(f"[WARN] {sp}/{cls}: 계획 {want}개 중 {got}개만 채웠습니다(예비 클립 소진).")
    if warned:
        print(f"[WARN] 전 창이 저 RMS 라 첫 후보를 그대로 쓴 클립 {len(warned)}개(품질 확인 권장):")
        for w, c in warned[:10]:
            print(f"  - {w} ({c})")
    print(f"provenance: {prov_path} ({len(prov) + len(new_rows)}행)")
    print("다음 단계: generate_dataset_index.py 재실행")


if __name__ == "__main__":
    main()
