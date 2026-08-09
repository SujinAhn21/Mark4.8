"""
chime_manifest.py — CHiME-Home 청크 annotation 을 읽어 chime_slicer 가 쓸 계획 CSV 를 만든다.

배경(2026-08-09): mark4.5 를 SINS 단일 출처로 재구축해 val accuracy 0.9953 을 얻었고, 세션 지문
검사까지 통과했다(학습에 안 쓴 세션 ex229·ex233 에서 0.9933·1.0000, others 확률의 세션 내부
sd 0.0300 > 세션 간 sd 0.0156). 그런데도 남는 문제가 두 가지 있었다.

  (1) 환경이 한 채뿐이다. SINS 는 한 사람이 vacation home 한 채에서 일주일 녹음한 것이 전부라,
      "다른 집에서도 되는가" 를 이 데이터로는 물을 수 없다.
  (2) SINS 는 한 세션이 한 활동이다. 그래서 타겟(watching_tv)과 others 가 세션을 공유하는 것이
      원천적으로 불가능하고, 세션 지문 가능성이 구조적으로 남는다.

CHiME-Home 은 이 둘을 동시에 메운다. 다른 집(영국 단독주택 lounge)이고, **같은 세션 안에서 TV 가
켜졌다 꺼졌다 한다.** 즉 한 세션에서 타겟과 others 를 둘 다 뽑을 수 있다. 세션이 클래스를 알려주지
못하므로 (2) 가 사라진다.

무엇보다 두 출처를 **타겟과 others 양쪽에 같은 비율로** 넣으면 출처가 클래스 정보를 주지 못한다.
"4.5 만 왜 높은가 / 출처 때문에 맞힌 것 아닌가" 라는 질문이 구조적으로 성립하지 않게 된다.
mark4.5 와 mark5.0 을 같은 방식으로 구성해 4.5 가 특별한 버전이 되지 않게 하는 것이 목적이다.

─────────────────────────────────────────────────────────────────────────────
라벨 선정 기준 (2026-08-09 실측)

CHiME-Home 라벨 7종: c 아동말소리 / m 성인남성 / f 성인여성 / v 비디오게임·TV /
                     p 타격음 / b 광대역 가전음 / o 기타   (+ S 무음, U 판단불가)

  타겟(media_talking) = v 만 붙은 청크. c·f·m 이 함께 붙은 것은 뺀다.
      v+c 가 373개나 되는데, 이건 TV 소리와 실제 사람 말소리가 겹친 것이라 media_talking 으로
      보기 애매하다. 타겟을 깨끗하게 두는 쪽을 택했다.
  others = v 가 전혀 없고 c·f·m·p·b·o 중 하나 이상이 붙은 청크. S·U 단독은 뺀다.

refined(2,762개, annotator 2명 이상 합의) 만으로는 수량이 모자란다. 그래서 raw 에서도 뽑되
**v 에 대해 3명 만장일치**를 요구한다 — refined 의 기준(2명 이상)보다 엄격하다.

이렇게 해도 되는 근거: refined 에서 탈락한 3,375개의 v 라벨 일치도를 직접 세어보니

      3명 만장일치 v   1,330
      2명만 v            163
      1명만 v            256

로, **탈락 이유가 v 때문인 청크는 419개(12%)뿐**이었다. 나머지 88% 는 v 판단은 셋 다 같았는데
p·b·o 구분에서 갈려 빠진 것이다. 원논문도 "c, m, f, v 는 합의가 강하고(median 0.864) p, b, o, S
는 낮다" 고 적고 있다. 즉 raw 를 쓰는 것이 품질을 낮추는 게 아니라, v 에 한해 기준을 더 올리면서
수량을 두 배로 만드는 일이다.

  최종 후보     타겟 1,321 / others 2,309
  품질 게이트   check_waveform 통과 타겟 1,050(79.5%) / others 1,896(82.1%)
                (탈락은 대부분 진폭극소 peak<0.02 — CHiME 은 SINS 보다 소리가 작다)

─────────────────────────────────────────────────────────────────────────────
세션 배정 (게이트 통과 기준 실측)

    세션              타겟   others   타겟비율
    270110_1632        408      212     65.8%   균형
    210110_0739        159      297     34.9%   균형
    200110_1711        122       99     55.2%   균형
    230110_1036         72      203     26.2%   균형
    220110_0731         28      717      3.8%   others 편중
    200110_1601        260        6     97.7%   타겟 편중
    230110_1501          1      362      0.3%   others 편중

**val·test 에는 균형 세션만 넣는다.** 지름길 검출이 val·test 에서 이뤄지기 때문이다. 한 세션이
한 클래스로 쏠려 있으면 "세션을 외웠는가" 를 그 split 에서 물을 수 없다.
편중 세션은 train 보조로만 쓴다. train 에서 세션 단서를 배웠다면 균형 세션으로 이뤄진 val·test 에서
드러난다.

    train  270110_1632 + 220110_0731      타겟 436 / others 929
    val    210110_0739                    타겟 159 / others 297
    test   200110_1711 + 230110_1036      타겟 194 / others 302

split 을 세션으로 나누는 이유는 SINS 와 같다. 같은 세션은 연속 녹음이라 앞뒤 청크가 같은 상황을
담고 있어, 세션을 쪼개면 train 과 test 에 같은 순간이 들어간다.

사용 예:
  python preprocessing/chime_manifest.py \
      --chime_root ~/workspace/chime_home_raw/chime_home \
      --out preprocessing/chime_manifest.csv
"""
import os
import csv
import argparse
from collections import Counter, defaultdict

# 세션 → split. 위 주석의 근거로 2026-08-09 확정.
SESSION_SPLIT = {
    "CR_lounge_270110_1632": "train",
    "CR_lounge_220110_0731": "train",
    "CR_lounge_210110_0739": "val",
    "CR_lounge_200110_1711": "test",
    "CR_lounge_230110_1036": "test",
    # 아래 둘은 쓰지 않는다. 타겟 비율이 97.7% / 0.3% 로 세션이 곧 클래스가 되어,
    # train 에 넣어도 얻는 것보다 잃는 것이 크다.
    "CR_lounge_200110_1601": None,
    "CR_lounge_230110_1501": None,
}

SPEECH = {"c", "f", "m"}
LIST_FILES = [
    ("refined", "development_chunks_refined.csv"),
    ("refined", "evaluation_chunks_refined.csv"),
    ("raw", "development_chunks_raw.csv"),
    ("raw", "evaluation_chunks_raw.csv"),
]


def read_chunk_csv(path):
    d = {}
    for row in csv.reader(open(path, encoding="utf-8")):
        if len(row) >= 2:
            d[row[0]] = row[1]
    return d


def classify(meta, tier):
    """청크 하나를 (role, category) 로 판정. 후보가 아니면 None.

    tier='refined' 면 majorityvote 를, 'raw' 면 annotator 3명을 직접 본다.
    raw 의 타겟 기준은 v 3명 만장일치 + 3명 모두 c·f·m 을 안 붙임(refined 보다 엄격).
    """
    if tier == "refined":
        labs = set(meta.get("majorityvote", "") or "")
        if labs == {"v"}:
            return "target", "v"
        if "v" not in labs and labs and labs not in ({"S"}, {"U"}):
            return "others", "".join(sorted(labs - set("SU"))) or None
        return None

    ann = [set(meta.get("annotation_a%d" % i, "") or "") for i in (1, 2, 3)]
    n_v = sum(1 for a in ann if "v" in a)
    if n_v == 3 and all(not (a & SPEECH) for a in ann):
        return "target", "v"
    if n_v == 0 and all(a and a not in ({"S"}, {"U"}) for a in ann):
        vote = Counter()
        for a in ann:
            for l in a:
                vote[l] += 1
        maj = "".join(sorted(l for l, c in vote.items() if c >= 2 and l not in "SU"))
        if maj:
            return "others", maj
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chime_root", type=str, required=True,
                    help="CHiME-Home 압축을 푼 폴더(안에 chunks/ 와 *_chunks_*.csv 가 있어야 한다)")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    root = os.path.abspath(os.path.expanduser(args.chime_root))
    chunk_dir = os.path.join(root, "chunks")
    if not os.path.isdir(chunk_dir):
        raise SystemExit(f"[ERROR] chunks 폴더가 없습니다: {chunk_dir}")

    # 목록 파일을 refined 먼저 읽어, 같은 청크가 raw 에도 있으면 refined 판정을 유지한다.
    seen, rows = set(), []
    tier_count = Counter()
    for tier, lf in LIST_FILES:
        path = os.path.join(root, lf)
        if not os.path.exists(path):
            raise SystemExit(f"[ERROR] 목록 파일이 없습니다: {path}")
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line or "," not in line:
                continue
            name = line.split(",", 1)[1]
            if name in seen:
                continue
            seen.add(name)
            sess = name.split(".s")[0]
            split = SESSION_SPLIT.get(sess, None)
            if split is None:
                continue
            got = classify(read_chunk_csv(os.path.join(chunk_dir, name + ".csv")), tier)
            if got is None:
                continue
            role, cat = got
            if not cat:
                continue
            wav = name + ".16kHz.wav"
            if not os.path.exists(os.path.join(chunk_dir, wav)):
                continue
            tier_count[(tier, role)] += 1
            rows.append({
                "split": split, "role": role, "wav": wav,
                "activity": cat, "sess": sess.replace("CR_lounge_", ""),
                "idx": name.rsplit("chunk", 1)[1], "node": tier,
            })

    if not rows:
        raise SystemExit("[ERROR] 후보가 하나도 없습니다.")
    rows.sort(key=lambda r: (r["split"], r["role"], r["sess"], int(r["idx"])))
    with open(os.path.abspath(os.path.expanduser(args.out)), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "role", "wav", "activity",
                                          "sess", "idx", "node"])
        w.writeheader()
        w.writerows(rows)

    print(f"[완료] {args.out} — {len(rows)}행")
    print("[출처별] " + ", ".join(f"{t}/{r} {n}" for (t, r), n in sorted(tier_count.items())))
    per = defaultdict(int)
    acts = defaultdict(Counter)
    for r in rows:
        per[(r["split"], r["role"])] += 1
        acts[(r["split"], r["role"])][r["activity"]] += 1
    print("\n[split × role]")
    for k in sorted(per):
        top = ", ".join(f"{a} {c}" for a, c in acts[k].most_common(6))
        print(f"  {k[0]:5s}/{k[1]:6s} {per[k]:4d}   [{top}]")
    print("\n[세션별]")
    ss = defaultdict(lambda: [0, 0])
    for r in rows:
        ss[(r["split"], r["sess"])][0 if r["role"] == "target" else 1] += 1
    for k in sorted(ss):
        t, o = ss[k]
        ratio = t / (t + o) * 100 if (t + o) else 0
        print(f"  {k[0]:5s} {k[1]:14s} 타겟 {t:4d} / others {o:4d}   타겟비율 {ratio:5.1f}%")


if __name__ == "__main__":
    main()
