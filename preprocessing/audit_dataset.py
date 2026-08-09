"""
audit_dataset.py — 학습 전에 데이터 **구성**을 점검하는 게이트.

validate_dataset.py 가 파일 하나하나의 품질(길이·rms·peak·클리핑)을 본다면, 이 스크립트는
**데이터셋 전체의 짜임새**를 본다. 파일은 전부 멀쩡한데 구성이 잘못돼서 학습이 헛도는 경우를
잡는 것이 목적이다.

■ 왜 만들었나 (2026-08-09)

mark4.5·mark5.0 을 만들면서 같은 성격의 사고가 세 번 났고, 셋 다 **파일 품질은 전부 정상**이었다.
그리고 셋 다 데이터만 보면 학습 전에 알 수 있었는데, 두 번은 수집 직후에 눈으로 발견했고
한 번은 **학습을 끝까지 돌리고 test 까지 본 뒤에야** 알았다.

  (1) test 세션 하나가 타겟 0 / others 55 가 됨 — 그 세션이 곧 클래스가 됨
      원인: 조합별 쿼터가 세션을 고려하지 않는데 타겟은 조합이 하나뿐이라 세션 배분이 안 일어남
  (2) heavy_impact 319개가 어른발걸음 171 / 아이들발걸음 74 / 망치질 74 로 쏠림(균등이면 106씩)
      원인: 이미 쓴 클립을 거르기 전에 카테고리 쿼터를 계산해 예비 목록의 앞 카테고리만 소진됨
  (3) ★train 타겟 250개 중 222개(89%)가 한 세션에서 나옴
      결과: CHiME 쪽 성능이 val 0.9720 -> test 0.7963 (17.6%p 하락), 재학습 필요
      원인: 세션 배정을 val·test 의 클래스 균형만 보고 정하고 train 의 세션 수는 안 봄

(3) 을 학습 전에 잡았다면 재학습 한 번을 아꼈다. 그래서 이 셋을 포함해 지금까지 겪은 구성
결함을 전부 항목으로 만들어 둔다. **수집 -> validate_dataset -> audit_dataset 순으로 돌리고,
audit 이 FAIL 없이 통과해야 학습으로 넘어간다.**

■ 점검 항목

  ① 출처 × 클래스        출처가 클래스를 알려주는가(출처만으로 맞힐 수 있는 비율)
  ② 세션 겹침            train/val/test 가 같은 세션을 공유하는가
  ③ val·test 세션 균형   한 세션이 한 클래스로 쏠렸는가(쏠리면 그 split 에서 지름길을 검출 못 함)
  ④ train 세션 다양성    클래스마다 몇 개 세션에서 왔는가, 최대 세션이 몇 %인가   <- (3) 을 잡는 항목
  ⑤ 카테고리 균등도      클래스 안에서 한 카테고리가 과대 대표되는가              <- (2) 를 잡는 항목
  ⑥ 증강본 격리          val·test 에 증강본이 섞이지 않았는가
  ⑦ 정합성               디스크 wav 수와 provenance active 행 수가 맞는가

■ 세션이란

  sins        aihub_src_file 의 ex번호(DevNode1_ex227_5.wav -> ex227). 한 세션 = 한 활동의 연속 녹음
  chime_home  aihub_volume(200110_1711). 한 세션 = 한 번의 연속 녹음(TV 를 켰다 껐다 함)
  aihub       세션 개념이 없다(클립이 독립). 세션 항목에서는 제외하고 ⑤ 카테고리로만 본다

■ 사용 예

  python preprocessing/audit_dataset.py --mark_version mark4.5
  python preprocessing/audit_dataset.py --mark_version mark5.0 \
      --data_root . --splits data_source_labeled,data_source_val,data_source_test
"""
import os
import re
import sys
import argparse
from collections import defaultdict, Counter

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# 폴더명 -> provenance 의 assigned_split 값(mark5 는 폴더명이 다르다)
DIR2SPLIT = {"data_source_labeled": "train", "data_source_val": "val",
             "data_source_test": "test", "train": "train", "val": "val", "test": "test"}

# 임계값 — 넘으면 WARN, 크게 넘으면 FAIL
TH = {
    "source_gap": 0.10,        # ① 두 클래스의 출처 비율 차이
    "sess_balance_warn": 0.15,  # ③ val·test 세션의 클래스 비율이 이 밖이면 WARN
    "sess_share_warn": 0.70,   # ④ train 에서 한 세션이 차지하는 비중
    "sess_share_fail": 0.85,
    "cat_ratio_warn": 3.0,     # ⑤ 클래스 안 카테고리 최대/최소 비
}

results = []


def report(level, item, msg):
    results.append((level, item, msg))
    mark = {"OK": "  ", "WARN": "⚠ ", "FAIL": "✗ "}[level]
    print(f"{mark}[{level:4s}] {item} — {msg}")


def sess_of(row):
    """세션 식별자. AI Hub 는 세션 개념이 없어 None."""
    src = str(row.get("source", ""))
    if src == "sins":
        m = re.search(r"ex(\d+)", str(row.get("aihub_src_file", "")))
        return "ex" + m.group(1) if m else None
    if src == "chime_home":
        v = row.get("aihub_volume")
        return str(v) if pd.notna(v) else None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mark_version", type=str, required=True)
    ap.add_argument("--data_root", type=str, default=os.path.join(PROJECT_ROOT, "data"))
    ap.add_argument("--splits", type=str, default="train,val,test")
    ap.add_argument("--provenance_path", type=str,
                    default=os.path.join(PROJECT_ROOT, "..", "data_provenance.xlsx"))
    args = ap.parse_args()

    prov = pd.read_excel(os.path.abspath(args.provenance_path))
    m = prov[prov["mark_version"] == args.mark_version]
    if "removed_20260715" in m.columns:
        m = m[m["removed_20260715"] == "active"]
    if m.empty:
        print(f"[ERROR] provenance 에 {args.mark_version} active 행이 없습니다.")
        sys.exit(1)
    orig = m[m["source_type"] == "original"] if "source_type" in m.columns else m
    aug = m[m["source_type"] != "original"] if "source_type" in m.columns else m.iloc[0:0]
    classes = sorted(m["target_class"].dropna().unique())
    print(f"\n{'='*78}\n[감사] {args.mark_version} — active {len(m)}행 "
          f"(원본 {len(orig)} / 증강 {len(aug)}), 클래스 {len(classes)}개\n{'='*78}\n")

    # ---------- ① 출처 × 클래스 ----------
    print("① 출처가 클래스를 알려주는가")
    if "source" in m.columns:
        for sp in ("train", "val", "test"):
            s = m[m["assigned_split"] == sp]
            if s.empty:
                continue
            tab = s.groupby(["source", "target_class"]).size().unstack(fill_value=0)
            for src in tab.index:
                row = tab.loc[src]
                tot = row.sum()
                top_cls = row.idxmax()
                share = row.max() / tot
                base = 1.0 / len(classes)
                lvl = "OK" if share <= base * 1.5 else ("WARN" if share < 0.9 else "FAIL")
                report(lvl, f"①{sp}/{src}",
                       f"출처만 알 때 최빈 클래스는 {top_cls} {share*100:.1f}% "
                       f"(균등이면 {base*100:.1f}%, n={tot})")
    else:
        report("OK", "①", "source 컬럼 없음 — 단일 출처로 간주")

    # ---------- ② 세션 겹침 ----------
    print("\n② 세션 겹침 (train/val/test 가 같은 세션을 공유하는가)")
    sess_map = defaultdict(lambda: defaultdict(set))
    for _, r in orig.iterrows():
        s = sess_of(r)
        if s:
            sess_map[str(r["source"])][r["assigned_split"]].add(s)
    if not sess_map:
        report("OK", "②", "세션 개념이 있는 출처 없음(AI Hub 단일)")
    for src, d in sorted(sess_map.items()):
        bad = []
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            ov = d.get(a, set()) & d.get(b, set())
            if ov:
                bad.append(f"{a}∩{b}={sorted(ov)}")
        report("FAIL" if bad else "OK", f"②{src}",
               "; ".join(bad) if bad else
               f"겹침 없음 (train {len(d.get('train',()))} / val {len(d.get('val',()))} / "
               f"test {len(d.get('test',()))} 세션)")

    # ---------- ③ val·test 세션 균형 ----------
    # 출처마다 '한 세션에 여러 클래스가 있을 수 있는가'가 다르다. SINS 는 한 세션이 한 활동이라
    # 세션마다 한 클래스만 나오는 것이 **구조적으로 정상**이고 고칠 수 없다. CHiME-Home 은 같은
    # 세션 안에서 TV 가 켜졌다 꺼졌다 하므로 두 클래스가 함께 나올 수 있다. 그래서 출처별로
    # 데이터에서 직접 판정한 뒤, 섞일 수 있는 출처에만 이 검사를 적용한다.
    print("\n③ val·test 세션이 한 클래스로 쏠렸는가")
    mixable = set()
    all_sess = defaultdict(Counter)
    for _, r in orig.iterrows():
        s = sess_of(r)
        if s:
            all_sess[(str(r["source"]), s)][r["target_class"]] += 1
    for (src, _), c in all_sess.items():
        if len(c) > 1:
            mixable.add(src)
    skipped = {src for src, _ in all_sess} - mixable
    for src in sorted(skipped):
        report("OK", f"③{src}",
               "한 세션에 한 클래스만 있는 출처 — 구조적 특성이라 검사 제외"
               " (검출은 섞일 수 있는 출처에서 한다)")
    any3 = False
    for sp in ("val", "test"):
        g = defaultdict(Counter)
        for _, r in orig[orig["assigned_split"] == sp].iterrows():
            s = sess_of(r)
            if s and str(r["source"]) in mixable:
                g[(str(r["source"]), s)][r["target_class"]] += 1
        for k, c in sorted(g.items()):
            any3 = True
            tot = sum(c.values())
            share = max(c.values()) / tot
            hi = 1 - TH["sess_balance_warn"]
            if len(c) == 1:
                lvl = "FAIL"
            elif share > hi:
                lvl = "WARN"
            else:
                lvl = "OK"
            report(lvl, f"③{sp}/{k[0]}/{k[1]}",
                   f"n={tot} " + ", ".join(f"{a} {b}" for a, b in c.most_common())
                   + f"  (최빈 {share*100:.1f}%)")
    if not any3 and not skipped:
        report("OK", "③", "세션 개념이 있는 출처 없음")

    # ---------- ④ train 세션 다양성 ----------
    print("\n④ train 세션 다양성 (한 세션에 몰려 있지 않은가)")
    any4 = False
    tr = orig[orig["assigned_split"] == "train"]
    for src in sorted(set(str(x) for x in tr.get("source", pd.Series(dtype=str)).dropna())):
        for cls in classes:
            sub = tr[(tr["source"] == src) & (tr["target_class"] == cls)]
            if sub.empty:
                continue
            c = Counter(s for s in (sess_of(r) for _, r in sub.iterrows()) if s)
            if not c:
                continue
            any4 = True
            tot = sum(c.values())
            share = max(c.values()) / tot
            lvl = ("FAIL" if share > TH["sess_share_fail"]
                   else "WARN" if share > TH["sess_share_warn"] else "OK")
            report(lvl, f"④{src}/{cls}",
                   f"세션 {len(c)}개, n={tot}, 최대 {c.most_common(1)[0][0]} "
                   f"{max(c.values())}개({share*100:.1f}%)  {dict(c.most_common())}")
    if not any4:
        report("OK", "④", "세션 개념이 있는 출처 없음")

    # ---------- ⑤ 카테고리 균등도 ----------
    print("\n⑤ 클래스 안 카테고리 균등도")
    if "aihub_category" in orig.columns:
        for cls in classes:
            sub = orig[orig["target_class"] == cls]
            c = Counter(str(x) for x in sub["aihub_category"].dropna())
            if len(c) < 2:
                continue
            ratio = max(c.values()) / max(1, min(c.values()))
            top = ", ".join(f"{a.split('_')[-1][:14]} {b}" for a, b in c.most_common(3))
            if cls == "others":
                # others 는 '타겟이 아닌 모든 소리'라 여러 출처·카테고리를 일부러 섞는다.
                # 균등을 요구하면 설계 의도와 어긋나므로 수치만 남기고 판정하지 않는다.
                report("OK", f"⑤{cls}",
                       f"카테고리 {len(c)}종 (다양성이 목적이라 균등 검사 제외)  [{top}]")
                continue
            lvl = "WARN" if ratio > TH["cat_ratio_warn"] else "OK"
            report(lvl, f"⑤{cls}",
                   f"카테고리 {len(c)}종, 최대/최소 {ratio:.1f}배  [{top}]")
    else:
        report("OK", "⑤", "aihub_category 컬럼 없음")

    # ---------- ⑥ 증강본 격리 ----------
    print("\n⑥ 증강본이 val·test 에 섞였는가")
    if len(aug):
        bad = aug[aug["assigned_split"].isin(["val", "test"])]
        report("FAIL" if len(bad) else "OK", "⑥",
               f"증강 {len(aug)}개 중 val·test 에 {len(bad)}개"
               + (f" — {sorted(bad['local_filename'].head(5))}" if len(bad) else ""))
    else:
        report("OK", "⑥", "증강본 없음")

    # ---------- ⑦ 정합성 ----------
    print("\n⑦ 디스크 vs provenance")
    root = os.path.abspath(args.data_root)
    disk = 0
    per_split = Counter()
    for sp in args.splits.split(","):
        d = os.path.join(root, sp.strip())
        if not os.path.isdir(d):
            continue
        for dirpath, _, fns in os.walk(d):
            n = sum(1 for f in fns if f.lower().endswith(".wav"))
            disk += n
            per_split[DIR2SPLIT.get(sp.strip(), sp.strip())] += n
    prov_per = Counter(m["assigned_split"])
    ok = disk == len(m) and all(per_split[k] == prov_per.get(k, 0) for k in per_split)
    report("OK" if ok else "FAIL", "⑦",
           f"디스크 {disk} / provenance active {len(m)}  "
           + ", ".join(f"{k} {per_split[k]}vs{prov_per.get(k,0)}" for k in sorted(per_split)))

    # ---------- 총평 ----------
    n_fail = sum(1 for l, _, _ in results if l == "FAIL")
    n_warn = sum(1 for l, _, _ in results if l == "WARN")
    print(f"\n{'='*78}")
    print(f"판정: OK {len(results)-n_fail-n_warn} / WARN {n_warn} / FAIL {n_fail}")
    if n_fail:
        print("✗ FAIL 이 있습니다. 구성을 고치고 다시 수집한 뒤 학습하십시오.")
        for l, i, msg in results:
            if l == "FAIL":
                print(f"   - {i}: {msg}")
    elif n_warn:
        print("⚠ FAIL 은 없습니다. WARN 은 의도한 것인지 확인한 뒤 진행하십시오.")
    else:
        print("문제 없음 — 학습으로 넘어가도 됩니다.")
    print("=" * 78)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
