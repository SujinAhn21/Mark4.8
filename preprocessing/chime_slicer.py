"""
chime_slicer.py — CHiME-Home 의 4초 청크에서 3초 세그먼트를 뽑는 스크립트.

`sins_slicer.py` 와 짝이다. mark4.5(그리고 mark5.0)의 media_talking / others 를 **두 집에서
절반씩** 뽑기 위해, SINS 쪽은 sins_slicer 가, CHiME-Home 쪽은 이 스크립트가 담당한다.
두 번 나눠 실행하는 이유는 원본 규격이 달라서다(SINS 10초 4채널 48kHz→16kHz / CHiME 4초 mono 16kHz).
provenance 에 이어붙이는 구조이고 next_index 가 provenance 를 보므로 파일 번호는 자연히 이어진다.

왜 두 출처를 섞는가(2026-08-09):
    SINS 단일 출처로도 채널 지름길은 잡혔지만(val 0.9953, 세션 내부 sd 0.0300 > 세션 간 0.0156),
    환경이 한 채뿐이고 SINS 는 한 세션이 한 활동이라 타겟과 others 가 세션을 공유할 수 없었다.
    CHiME-Home 은 **같은 세션 안에서 TV 가 켜졌다 꺼졌다 하므로** 그 구멍을 메운다.
    그리고 두 출처가 타겟과 others 양쪽에 같은 비율로 들어가면 출처가 클래스 정보를 주지 못한다.
    선정 근거·라벨 기준·세션 배정은 `chime_manifest.py` 머리주석에 실측값과 함께 적어두었다.

원본 구조가 sins 와 다른 점:
  - wav 가 split/role 폴더로 나뉘어 있지 않고 `chunks/` 한 곳에 전부 있다. 어느 split·role 인지는
    매니페스트가 정한다(--chunk_dir 하나만 받는 이유).
  - 4초짜리라 3초를 빼면 여유가 1초뿐이다. 창은 가운데(0.5~3.5s) 고정, 떨어지면 앞(0.0)→뒤(1.0)를
    시도하지만 겹침이 커서 실질적으로는 다음 파일로 넘어간다고 보는 편이 맞다.
  - 이미 16kHz mono 라 리샘플·채널선택이 필요 없다.
  - DC 오프셋이 |1e-5| 수준으로 사실상 0 이다(SINS 는 노드마다 최대 0.043). 그래도 sins 와 같은
    처리를 통과시키려고 창 평균을 빼는 것은 그대로 둔다. 빼도 값이 거의 안 바뀐다.

품질 게이트는 sins 와 같은 validate_dataset.check_waveform 을 쓴다. 2026-08-09 전수 측정 결과
후보 타겟 1,321개 중 1,050개(79.5%), others 2,309개 중 1,896개(82.1%)가 통과했다. 탈락은 대부분
`진폭극소 peak<0.02` 다 — CHiME 은 SINS 보다 녹음 레벨이 낮다.

provenance 컬럼은 aihub_slicer·sins_slicer 와 같은 이름을 재사용한다(컬럼을 새로 만들면
resplit·validate·augment 가 읽지 못한다). 구분은 source 값(chime_home)으로 한다.
    aihub_src_file = 청크 wav 이름(CR_lounge_270110_1632.s0_chunk12.16kHz.wav)
    aihub_category = 라벨 조합(v / cp / cop …)
    aihub_volume   = 세션명(270110_1632)

⚠ resplit_dataset.py --stratify_source 를 돌리면 안 된다. 세션 분리가 깨진다.

사용 예:
  python preprocessing/chime_slicer.py --mark_version mark4.5 \
      --chunk_dir ~/workspace/chime_home_raw/chime_home/chunks \
      --manifest preprocessing/chime_manifest.csv --dry_run
"""
import os
import re
import sys
import csv
import time
import shutil
import hashlib
import argparse
from datetime import date
from collections import defaultdict

import numpy as np
import pandas as pd
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from validate_dataset import check_waveform                      # noqa: E402
# aihub_slicer 에서 import 하지 않는 이유: 그 모듈은 최상위에서 torch 를 import 하는데
# /mnt/c 의 venv 에서는 그것만 76초가 걸린다(2026-07-31 실측).

TARGET_SR = 16000
SEG_SEC = 3.0
SRC_SEC = 4.0
# 4초에서 3초를 뺀 1초를 반으로 나눈 0.5 가 가운데.
WINDOW_STARTS = (0.5, 0.0, 1.0)


def target_class_from_config(mark_version):
    """vild_config.py 에서 이 버전의 타겟 클래스명을 읽는다(mark4.5 -> media_talking).

    소스를 텍스트로 읽어 정규식으로 뽑는다(torch 를 끌고 오지 않으려고). 손으로 적지 않는 이유는
    generate_dataset_index 의 키워드 맵이 같은 이름을 쓰기 때문이다. 어긋나면 인덱스에서 빠진다.
    """
    cfg_path = os.path.join(PROJECT_ROOT, "vild", "vild_config.py")
    src = open(cfg_path, encoding="utf-8").read()
    m = re.search(
        r'mark_version\s*==\s*["\']%s["\'][^\n]*\n\s*self\.classes\s*=\s*\[\s*["\']([^"\']+)["\']'
        % re.escape(mark_version), src)
    if not m:
        raise SystemExit(f"[ERROR] {cfg_path} 에서 '{mark_version}' 클래스 정의를 못 찾았습니다. "
                         f"--target_class 로 직접 지정하십시오.")
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
    """이어붙일 다음 번호. 디스크뿐 아니라 provenance 에 기록된 이름(제거된 행 포함)까지 본다."""
    disk = os.listdir(data_split_dir) if os.path.isdir(data_split_dir) else []
    return max(_max_index(disk, prefix), _max_index(prov_names, prefix)) + 1


def read_window(path, start_sec, remove_dc=True):
    """청크 wav 에서 start_sec 부터 3초를 float32 모노로 읽는다.

    CHiME-Home 16kHz 판은 이미 mono 라 채널 선택이 필요 없지만, 48kHz 판을 잘못 넘겼을 때를
    대비해 always_2d 로 읽고 첫 채널만 쓴다. 샘플레이트가 다르면 예외로 막는다(리샘플하지 않는다 —
    다른 버전과 전처리를 같게 두려면 원본이 16kHz 여야 한다).
    """
    info = sf.info(path)
    if info.samplerate != TARGET_SR:
        raise ValueError(f"샘플레이트 {info.samplerate} != {TARGET_SR} (16kHz 판을 쓰십시오)")
    beg = int(round(start_sec * TARGET_SR))
    end = beg + int(round(SEG_SEC * TARGET_SR))
    if end > info.frames:
        raise ValueError(f"길이부족 frames={info.frames} < {end}")
    x, _ = sf.read(path, start=beg, stop=end, dtype="float32", always_2d=True)
    seg = np.ascontiguousarray(x[:, 0])
    if remove_dc:
        seg = seg - np.float32(seg.mean())
    return seg


def load_manifest(path):
    """계획 CSV → split/role 별 목록. 열: split,role,wav,activity,sess,idx,node"""
    need = {"split", "role", "wav", "activity", "sess", "idx", "node"}
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        raise SystemExit(f"[ERROR] manifest 가 비어 있습니다: {path}")
    missing = need - set(rows[0])
    if missing:
        raise SystemExit(f"[ERROR] manifest 에 없는 열: {sorted(missing)}")
    out = defaultdict(list)
    for r in rows:
        out[(r["split"], r["role"])].append(r)
    return out


def activity_quota(items, want):
    """칸 하나의 필요 개수를 라벨 조합별로 쪼갠다. 비율은 매니페스트의 조합별 개수 비율 그대로.

    sins_slicer 와 같은 장치다. 목록을 앞에서부터 꺼내 쓰면 앞쪽 조합에서 쿼터가 다 차서
    others 가 한두 조합으로 쏠린다(2026-08-08 SINS 1차 실행 때 vacuum_cleaner 4개·eating 0개가
    나왔던 사고). CHiME others 는 cp·c·cf·cop·b·op 등 조합이 다양한데, 이 다양성이 others 의
    역할이므로 비율을 지켜야 한다.

    끝수는 최대잉여법으로 배분해 합이 정확히 want 가 되게 한다.
    """
    counts = defaultdict(int)
    for r in items:
        counts[r["activity"]] += 1
    total = sum(counts.values())
    if total == 0:
        return {}
    raw = {a: want * c / total for a, c in counts.items()}
    quota = {a: int(v) for a, v in raw.items()}
    rest = want - sum(quota.values())
    for a in sorted(raw, key=lambda k: (-(raw[k] - quota[k]), k))[:rest]:
        quota[a] += 1
    return quota


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mark_version", type=str, required=True)
    ap.add_argument("--chunk_dir", type=str, required=True,
                    help="CHiME-Home chunks 폴더(*.16kHz.wav 가 전부 여기 있다)")
    ap.add_argument("--manifest", type=str, required=True,
                    help="chime_manifest.py 가 만든 계획 CSV")
    ap.add_argument("--provenance_path", type=str,
                    default=os.path.join(PROJECT_ROOT, "..", "data_provenance.xlsx"))
    ap.add_argument("--target_class", type=str, default=None,
                    help="타겟 클래스명. 생략하면 vild_config.py 에서 읽는다")
    ap.add_argument("--source_name", type=str, default="chime_home")
    # 기본값은 mark4.5 를 SINS 와 절반씩 나눌 때의 CHiME 몫이다.
    # 전체 500/215/215 중 SINS 250/108/107 + CHiME 250/107/108.
    ap.add_argument("--target_train", type=int, default=250)
    ap.add_argument("--target_val", type=int, default=107)
    ap.add_argument("--target_test", type=int, default=108)
    ap.add_argument("--others_train", type=int, default=250)
    ap.add_argument("--others_val", type=int, default=107)
    ap.add_argument("--others_test", type=int, default=108)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--no_quality_check", action="store_true",
                    help="잘라낸 3초의 품질 검사를 끈다(권장하지 않음)")
    ap.add_argument("--keep_dc", action="store_true",
                    help="창 평균을 빼지 않는다. CHiME 은 DC 가 사실상 0 이라 차이가 거의 없다")
    args = ap.parse_args()

    prov_path = os.path.abspath(args.provenance_path)
    chunk_dir = os.path.abspath(os.path.expanduser(args.chunk_dir))
    target_class = args.target_class or target_class_from_config(args.mark_version)
    manifest = load_manifest(os.path.abspath(os.path.expanduser(args.manifest)))

    need = {
        ("train", "target"): args.target_train, ("val", "target"): args.target_val,
        ("test", "target"): args.target_test,
        ("train", "others"): args.others_train, ("val", "others"): args.others_val,
        ("test", "others"): args.others_test,
    }
    print(f"[INFO] mark_version={args.mark_version}  타겟 클래스={target_class}  "
          f"source={args.source_name}  창=가운데 고정 {WINDOW_STARTS[0]}s (대체 {WINDOW_STARTS[1:]})  "
          f"DC제거={'끔(--keep_dc)' if args.keep_dc else '켬'}")

    # 1) provenance 대조: 고아 파일 점검 + 이미 쓴 창 파악
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
            print("        정리한 뒤 다시 실행하십시오.")
            sys.exit(1)
    used_win = set()
    if {"aihub_src_file", "seg_start_sec"} <= set(mine.columns):
        got = mine.loc[mine["aihub_src_file"].notna(), ["aihub_src_file", "seg_start_sec"]]
        used_win = {(str(a), round(float(b), 3))
                    for a, b in got.itertuples(index=False, name=None) if pd.notna(b)}
    n_src = int((mine["source"] == args.source_name).sum()) if "source" in mine.columns else 0
    print(f"[INFO] provenance 전체 {len(prov)}행 중 {args.mark_version} {len(mine)}행"
          f"(active {len(active)}, 그중 {args.source_name} {n_src}), 이미 쓴 창 {len(used_win)}개")

    # 2) 원본 존재 확인 + 계획 점검
    plan, shortage = {}, []
    for (sp, role), items in sorted(manifest.items()):
        avail = [r for r in items if os.path.exists(os.path.join(chunk_dir, r["wav"]))]
        plan[(sp, role)] = avail
        n = need[(sp, role)]
        if len(avail) < n:
            shortage.append((sp, role, n, len(avail)))
        acts = defaultdict(int)
        sesses = defaultdict(int)
        for r in avail:
            acts[r["activity"]] += 1
            sesses[r["sess"]] += 1
        detail = ", ".join(f"{a} {c}" for a, c in
                           sorted(acts.items(), key=lambda kv: -kv[1])[:6])
        sdetail = ", ".join(f"{s} {c}" for s, c in sorted(sesses.items()))
        print(f"  {sp:5s}/{role:6s}: 필요 {n:3d}  원본 {len(avail):4d}개  "
              f"여유 {len(avail)/n if n else 0:.2f}배")
        print(f"          세션 [{sdetail}]")
        print(f"          조합 [{detail}]")
    if shortage:
        print("[ERROR] 원본이 부족합니다:")
        for sp, role, n, have in shortage:
            print(f"  - {sp}/{role}: 필요 {n}, 있음 {have}")
        sys.exit(1)
    if args.dry_run:
        print("[DRY RUN] 추출은 하지 않고 종료합니다.")
        return

    # 3) 추출 + 저장 + provenance
    backup = prov_path + f".bak_before_chime_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(prov_path, backup)
    print(f"[INFO] provenance 백업: {os.path.basename(backup)}")
    if "source" not in prov.columns:
        prov["source"] = "fsd50k"
    prov_names = set(prov["local_filename"].astype(str))

    counters, new_rows, rejected, unfilled = {}, [], [], []
    sess_used, act_used = defaultdict(int), defaultdict(int)
    t0 = time.time()
    todo = sum(need.values())
    done = 0

    def try_file(rec):
        """파일 하나를 창 순서(가운데→앞→뒤)대로 시도. 성공하면 (시작초, 파형), 실패면 None."""
        path = os.path.join(chunk_dir, rec["wav"])
        for start in WINDOW_STARTS:
            if (rec["wav"], round(start, 3)) in used_win:
                rejected.append((rec["wav"], start, rec["activity"], "이미쓴창"))
                continue
            try:
                seg = read_window(path, start, remove_dc=not args.keep_dc)
            except Exception as e:
                rejected.append((rec["wav"], start, rec["activity"], f"읽기실패 {e}"))
                continue
            bad = (None if args.no_quality_check
                   else check_waveform(seg, TARGET_SR, min_duration=SEG_SEC))
            if bad is None:
                return start, seg
            rejected.append((rec["wav"], start, rec["activity"], bad))
        return None

    for sp in ("val", "test", "train"):
        for role in ("target", "others"):
            cls = target_class if role == "target" else "others"
            data_dir = os.path.join(PROJECT_ROOT, "data", sp)
            os.makedirs(data_dir, exist_ok=True)
            key = (sp, cls)
            counters[key] = next_index(data_dir, f"{cls}_{sp}_", prov_names)

            want = need[(sp, role)]
            items = plan[(sp, role)]
            quota = activity_quota(items, want)
            by_act = defaultdict(list)
            for r in items:
                by_act[r["activity"]].append(r)
            # 조합별 쿼터만큼 먼저 뽑고, 못 채운 만큼은 여유 있는 조합에서 빌린다.
            # 조합 안에서는 **세션을 번갈아** 꺼낸다.
            #
            # [2026-08-09 수정] 처음에는 조합별로 앞에서부터 꺼내 썼는데, 타겟은 조합이 'v' 하나뿐이라
            # 세션 배분이 전혀 일어나지 않았다. 그 결과 test 타겟 108개가 전부 200110_1711 에서
            # 나오고 230110_1036 은 타겟 0 / others 55 가 되어, **그 세션이 others 전용이 됐다.**
            # 세션이 곧 클래스가 되면 "세션을 외웠는가" 를 test 에서 물을 수 없다. CHiME 을 넣은
            # 이유 자체가 SINS 가 못 하는 그것(한 세션 안에 두 클래스가 함께 있는 것)이므로,
            # 세션을 고르게 섞지 않으면 넣은 의미가 없어진다.
            queue, leftover = [], []
            for a in sorted(by_act):
                q = quota.get(a, 0)
                by_sess = defaultdict(list)
                for r in by_act[a]:
                    by_sess[r["sess"]].append(r)
                picked = []
                while len(picked) < q and any(by_sess.values()):
                    for s in sorted(by_sess):
                        if by_sess[s] and len(picked) < q:
                            picked.append(by_sess[s].pop(0))
                queue.extend(picked)
                leftover.extend(r for s in sorted(by_sess) for r in by_sess[s])
            print(f"  [{sp}/{role}] 조합 쿼터: "
                  + ", ".join(f"{a} {quota.get(a, 0)}" for a in sorted(by_act)
                              if quota.get(a, 0))
                  + f"  (예비 {len(leftover)})")

            made_here = 0
            while made_here < want and (queue or leftover):
                rec = queue.pop(0) if queue else leftover.pop(0)
                got = try_file(rec)
                if got is None:
                    continue                      # 다음 예비 파일로
                start, seg = got
                local_name = f"{cls}_{sp}_{counters[key]:03d}.wav"
                out_path = os.path.join(data_dir, local_name)
                sf.write(out_path, seg, TARGET_SR, subtype="FLOAT")
                counters[key] += 1
                made_here += 1
                done += 1
                used_win.add((rec["wav"], round(start, 3)))
                sess_used[rec["sess"]] += 1
                act_used[rec["activity"]] += 1
                new_rows.append({
                    "local_filename": local_name,
                    "original_labels": rec["activity"],
                    "target_class": cls, "assigned_split": sp,
                    "mark_version": args.mark_version,
                    "sha256": hashlib.sha256(open(out_path, "rb").read()).hexdigest(),
                    "size_bytes": os.path.getsize(out_path),
                    "download_date": date.today().isoformat(), "source_type": "original",
                    "removed_20260715": "active",
                    "source": args.source_name,
                    "aihub_src_file": rec["wav"],
                    "aihub_category": rec["activity"],
                    "aihub_volume": rec["sess"],
                    "seg_start_sec": round(start, 3),
                    "seg_end_sec": round(start + SEG_SEC, 3),
                })
                if done % 200 == 0:
                    merged = pd.concat([prov, pd.DataFrame(new_rows)], ignore_index=True)
                    merged.to_excel(prov_path, index=False)
                    print(f"[진행] {done}/{todo} — {(time.time()-t0)/60:.1f}분, provenance 중간 저장",
                          flush=True)
            if made_here < want:
                unfilled.append((sp, role, want, made_here))

    if new_rows:
        merged = pd.concat([prov, pd.DataFrame(new_rows)], ignore_index=True)
        merged.to_excel(prov_path, index=False)

    print(f"\n[완료] 신규 저장 {len(new_rows)}개 ({(time.time()-t0)/60:.1f}분)")
    made = defaultdict(int)
    for r in new_rows:
        made[(r["assigned_split"], r["target_class"])] += 1
    print("[실제] " + ", ".join(f"{sp}/{c} +{n}" for (sp, c), n in sorted(made.items())))
    print("[세션별 채택] " + ", ".join(f"{k} {v}" for k, v in sorted(sess_used.items())))
    print("[조합별 채택] " + ", ".join(f"{k} {v}" for k, v in
                                   sorted(act_used.items(), key=lambda kv: -kv[1])[:12]))
    if rejected:
        print(f"[품질탈락] {len(rejected)}회(다음 창이나 예비 파일로 대체):")
        by_act = defaultdict(int)
        for _, _, a, why in rejected:
            by_act[a] += 1
        print("  조합별: " + ", ".join(f"{k} {v}" for k, v in
                                    sorted(by_act.items(), key=lambda kv: -kv[1])[:10]))
        for f, s, a, why in rejected[:10]:
            print(f"  - {f} t={s}s ({a}): {why}")
    if unfilled:
        print("[WARN] 예비까지 소진해 못 채운 칸:")
        for sp, role, want, got in unfilled:
            print(f"  - {sp}/{role}: 필요 {want}, 채움 {got}")

    print("\n다음 단계: sins_slicer 몫까지 다 채운 뒤 augment_density.py --dry_run 으로"
          " 밀도부터 재고 모드를 정할 것 (원본 격차 0.05 초과면 기본, 이하면 --balanced).")


if __name__ == "__main__":
    main()
