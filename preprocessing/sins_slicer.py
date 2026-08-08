"""
sins_slicer.py — SINS(DCASE 2018 Task 5)의 10초 4채널 wav 묶음에서 3초 세그먼트를 뽑는 스크립트.

배경(2026-08-08): mark4.5 를 VOiCES 타겟 + AI Hub others 로 만들었더니 val accuracy 가 1.0000,
ROC AUC 1.0000 이 나왔다. 잘 된 것이 아니라 **녹음 채널이 클래스와 완전히 교란된 것**이었다.
provenance 와 대조해 실측한 근거:

    버전      타겟            others 확률 sd   카테고리 평균 폭   결정축 폭   내용이 쓰는 비율
    mark4.1  heavy_impact        0.1358          0.4276       0.8211        52.1%
    mark4.3  construction        0.1605          0.3127       0.8240        37.9%
    mark4.5  media_talking       0.0057          0.0132       0.7983         1.7%   <- 이상

    others 215개(AI Hub 37개 카테고리: 발걸음·망치질·세탁기·굴착기·바이올린·아이들 떠드는 소리…)가
    전부 Prob 0.108 근처 한 점으로 뭉쳤다(sd 0.0057, 절반이 0.003 폭 안). 내용을 듣고 있다면
    피아노와 굴착기가 같은 값을 받을 이유가 없다. 특히 '등하원 아이들 떠드는 소리'(실제 사람
    말소리)가 하위권이었다 — media_talking 의 본질이 사람 말소리인데 가장 안 헷갈렸다는 뜻이다.
    타겟(VOiCES) 최솟값 0.7037, others 최댓값 0.1603 으로 사이가 0.5434 통째로 비어 있었다.

그래서 타겟과 others 를 **같은 녹음 환경**에서 뽑기로 했다. SINS 는 한 사람이 한 집에서 일주일간
생활한 것을 집 안 마이크 어레이 4개(DevNode1~4)로 계속 녹음하고 활동별로 라벨을 붙인 데이터라,
watching_tv 와 cooking·dishwashing·vacuum_cleaner·social_activity·eating·working 이 전부 같은
마이크·같은 집·같은 녹음 체인에서 나온다. 채널로는 구분이 안 되므로 모델이 내용을 들어야 한다.

aihub_slicer / longwav_slicer 를 쓸 수 없는 이유:
  - aihub_slicer 는 클립마다 annotation json 이 붙어 "소리 구간"을 아는 구조를 전제한다.
  - longwav_slicer 는 55분짜리 긴 wav 하나를 시간축으로 잘라 split 을 나눈다.
  - SINS 는 10초짜리 4채널 wav 가 여러 개이고, split 은 이미 **세션 단위**로 정해져 있다.

split 을 세션으로 나누는 이유: 같은 시각을 노드 4개가 동시에 녹음한다. 노드로 나누면 train 의
그 순간과 test 의 그 순간이 같은 내용이 되어 누수가 난다(VOiCES 때 마이크 시간정렬 포락선 상관
0.256 대 대조군 0.005 로 확인했던 것과 같은 문제). 세션이 곧 서로 다른 녹음 시간이라, 세션을
통째로 한 split 에 넣으면 이 문제가 원천적으로 막힌다.

10초에서 3초를 어디서 자르는가: **가운데 고정(3.5~6.5초)**. 2026-08-08 수진님 결정.
떨어지면 앞(0~3초) → 뒤(7~10초) 순으로 시도하고, 그래도 안 되면 다음 예비 파일로 넘어간다.
"소리가 가장 활발한 구간"을 고르는 방식은 채택하지 않았다. 수집 단계에서 시끄러운 곳만 뽑으면
증강 모드를 정하는 원본 밀도 측정(2026-08-01 규칙)이 왜곡되기 때문이다.

4채널 중 1채널(ch0)만 쓴다. 나머지 7개 버전이 전부 모노이고, 어레이 안 마이크 4개는 5cm 간격이라
거의 같은 소리다. 채널을 섞으면 4.5만 조건이 달라진다.

품질 게이트는 validate_dataset.check_waveform 을 그대로 쓴다(수집 기준 = 사후 검증 기준).

provenance 컬럼은 aihub_slicer 와 같은 이름을 재사용한다. 컬럼을 새로 만들면 resplit·validate·
augment 가 읽지 못한다. 구분은 source 컬럼 값(sins)으로 한다.
    aihub_src_file = 원본 wav 이름(DevNode1_ex227_5.wav)
    aihub_category = SINS 활동명(watching_tv / cooking / …)
    aihub_volume   = 노드(DevNode1)

⚠ resplit_dataset.py --stratify_source 를 돌리면 안 된다. 클래스 안에서 파일을 무작위로 섞어
   split 을 다시 배정하므로 위 세션 분리가 깨진다.

사용 예:
  python preprocessing/sins_slicer.py --mark_version mark4.5 \
      --wav_root ~/workspace/sins_raw --manifest preprocessing/sins_manifest.csv --dry_run
  python preprocessing/sins_slicer.py --mark_version mark4.5 \
      --wav_root ~/workspace/sins_raw --manifest preprocessing/sins_manifest.csv
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
SRC_SEC = 10.0
# 가운데 → 앞 → 뒤 순으로 시도한다. 10초에서 3초를 뺀 7초를 반으로 나눈 3.5 가 가운데.
WINDOW_STARTS = (3.5, 0.0, 7.0)


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
    """wav 에서 start_sec 부터 3초를 ch0 만 읽어 float32 모노로 돌려준다.

    remove_dc: 창의 평균을 빼서 DC 오프셋을 없앤다(기본 켬).

    [2026-08-08 신설] SINS 는 마이크 어레이마다 고유한 DC 바이어스가 있다(12비트 MEMS,
    Sonion N8AC03). 1,860개 실측:

        노드         개수    DC 평균     노드 안 표준편차   |DC|/rms 평균
        DevNode1     470   -0.02969      0.00115           0.754
        DevNode2     468   +0.01556      0.00032           0.645
        DevNode3     457   -0.00403      0.00044           0.242
        DevNode4     465   +0.04274      0.00038           0.892

    노드 안에서 거의 상수라 하드웨어 특성이다. 신호 세기의 70~90%에 해당하는 직류 성분이
    얹혀 있어, 그대로 두면 멜 스펙트로그램 최저 대역에 큰 에너지로 들어가고 검증에서
    DC_OFFSET WARN 이 2,212건(전체의 77%) 났다.

    클래스와 상관되지는 않았다(media_talking 평균 0.00605 / others 0.00617, 노드도
    클래스마다 231~236 로 고르게 배정됨). 즉 지름길은 아니다. 그래도 빼는 이유는,
    소리가 아닌 성분이라 잃는 정보가 없고, 다른 7개 버전은 애초에 DC 가 거의 0 이라
    DC 를 빼야 4.5 가 나머지와 같은 조건이 되기 때문이다.
    (2026-08-02 에 채널 보정용 저역통과 필터를 "4.5만 다른 전처리를 받는 값을 못 한다"며
     기각한 것과는 다른 경우다. 그때는 신호를 깎아 정보를 잃는 쪽이었다.)
    """
    info = sf.info(path)
    if info.samplerate != TARGET_SR:
        raise ValueError(f"샘플레이트 {info.samplerate} != {TARGET_SR}")
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
    """다운로드 계획 CSV → split/role 별 파일 목록. 열: split,role,wav,activity,sess,idx,node"""
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
    """칸 하나의 필요 개수를 활동별로 쪼갠다. 비율은 manifest 의 활동별 개수 비율을 그대로 쓴다.

    [2026-08-08 수정] 처음에는 매니페스트를 앞에서부터 필요 개수만큼 꺼내 썼는데, 매니페스트가
    활동 순서대로 이어붙인 목록이라 30% 여유분 때문에 앞쪽 활동에서 쿼터가 다 차 버렸다.
    그 결과 others 930개가 social_activity 299 / working 250 / cooking 234 / dishwashing 143 /
    vacuum_cleaner 4 / eating 0 으로 쏠렸다(계획은 230/200/180/110/110/100). others 를 고르게
    짜서 청소기·식사 소리까지 넣으려던 설계가 무너지므로 활동별로 쿼터를 나눈다.

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
    ap.add_argument("--wav_root", type=str, required=True,
                    help="SINS 원본 루트. 아래에 {split}/{role}/*.wav 가 있어야 한다")
    ap.add_argument("--manifest", type=str, required=True,
                    help="다운로드 계획 CSV(split,role,wav,activity,sess,idx,node)")
    ap.add_argument("--provenance_path", type=str,
                    default=os.path.join(PROJECT_ROOT, "..", "data_provenance.xlsx"))
    ap.add_argument("--target_class", type=str, default=None,
                    help="타겟 클래스명. 생략하면 vild_config.py 에서 읽는다")
    ap.add_argument("--source_name", type=str, default="sins")
    ap.add_argument("--target_train", type=int, default=500)
    ap.add_argument("--target_val", type=int, default=215)
    ap.add_argument("--target_test", type=int, default=215)
    ap.add_argument("--others_train", type=int, default=500)
    ap.add_argument("--others_val", type=int, default=215)
    ap.add_argument("--others_test", type=int, default=215)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--no_quality_check", action="store_true",
                    help="잘라낸 3초의 품질 검사를 끈다(권장하지 않음)")
    ap.add_argument("--keep_dc", action="store_true",
                    help="DC 오프셋을 빼지 않고 원본 그대로 저장한다(권장하지 않음). "
                         "기본은 제거 — SINS 는 노드마다 고유 DC 바이어스가 있다")
    args = ap.parse_args()

    prov_path = os.path.abspath(args.provenance_path)
    wav_root = os.path.abspath(os.path.expanduser(args.wav_root))
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
    print(f"[INFO] provenance 전체 {len(prov)}행 중 {args.mark_version} {len(mine)}행"
          f"(active {len(active)}), 이미 쓴 창 {len(used_win)}개")

    # 2) 원본 존재 확인 + 계획 점검
    plan, shortage = {}, []
    for (sp, role), items in sorted(manifest.items()):
        d = os.path.join(wav_root, sp, role)
        avail = [r for r in items if os.path.exists(os.path.join(d, r["wav"]))]
        plan[(sp, role)] = avail
        n = need[(sp, role)]
        if len(avail) < n:
            shortage.append((sp, role, n, len(avail)))
        acts = defaultdict(int)
        for r in avail:
            acts[r["activity"]] += 1
        detail = ", ".join(f"{a} {c}" for a, c in sorted(acts.items(), key=lambda kv: -kv[1]))
        print(f"  {sp:5s}/{role:6s}: 필요 {n:3d}  원본 {len(avail):3d}개"
              f"(계획 {len(items)})  [{detail}]")
    if shortage:
        print("[ERROR] 원본이 부족합니다:")
        for sp, role, n, have in shortage:
            print(f"  - {sp}/{role}: 필요 {n}, 있음 {have}")
        sys.exit(1)
    if args.dry_run:
        print("[DRY RUN] 추출은 하지 않고 종료합니다.")
        return

    # 3) 추출 + 저장 + provenance
    backup = prov_path + f".bak_before_sins_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(prov_path, backup)
    print(f"[INFO] provenance 백업: {os.path.basename(backup)}")
    if "source" not in prov.columns:
        prov["source"] = "fsd50k"
    prov_names = set(prov["local_filename"].astype(str))

    counters, new_rows, rejected, unfilled = {}, [], [], []
    node_used, act_used = defaultdict(int), defaultdict(int)
    t0 = time.time()
    todo = sum(need.values())
    done = 0

    def try_file(rec, d):
        """파일 하나를 창 순서(가운데→앞→뒤)대로 시도. 성공하면 (시작초, 파형), 실패면 None."""
        path = os.path.join(d, rec["wav"])
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
            d = os.path.join(wav_root, sp, role)
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
            # 활동별 쿼터만큼 먼저 뽑고, 못 채운 만큼은 여유 있는 활동에서 빌린다.
            queue, leftover = [], []
            for a in sorted(by_act):
                q = quota.get(a, 0)
                queue.extend(by_act[a][:q])
                leftover.extend(by_act[a][q:])
            print(f"  [{sp}/{role}] 활동별 쿼터: "
                  + ", ".join(f"{a} {quota.get(a, 0)}" for a in sorted(by_act))
                  + f"  (예비 {len(leftover)})")

            made_here = 0
            while made_here < want and (queue or leftover):
                rec = queue.pop(0) if queue else leftover.pop(0)
                got = try_file(rec, d)
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
                node_used[rec["node"]] += 1
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
                    "aihub_volume": f"DevNode{rec['node']}",
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
    print("[노드별 채택] " + ", ".join(f"DevNode{k} {v}" for k, v in sorted(node_used.items())))
    print("[활동별 채택] " + ", ".join(f"{k} {v}" for k, v in
                                   sorted(act_used.items(), key=lambda kv: -kv[1])))
    if rejected:
        print(f"[품질탈락] {len(rejected)}회(다음 창이나 예비 파일로 대체):")
        by_act = defaultdict(int)
        for _, _, a, why in rejected:
            by_act[a] += 1
        print("  활동별: " + ", ".join(f"{k} {v}" for k, v in
                                    sorted(by_act.items(), key=lambda kv: -kv[1])))
        for f, s, a, why in rejected[:10]:
            print(f"  - {f} t={s}s ({a}): {why}")
    if unfilled:
        print("[WARN] 예비까지 소진해 못 채운 칸:")
        for sp, role, want, got in unfilled:
            print(f"  - {sp}/{role}: 필요 {want}, 채움 {got}")

    print("\n다음 단계: augment_density.py --dry_run 으로 밀도부터 재고 모드를 정할 것"
          " (원본 격차 0.05 초과면 기본, 이하면 --balanced).")


if __name__ == "__main__":
    main()
