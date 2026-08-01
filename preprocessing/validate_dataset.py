"""
validate_dataset.py — 데이터셋 유효성 검증(4.x 시리즈 공용).

배경(2026-07-31): mark4.8 재구성 데이터 2860개 중 3초 미만이 206개(7.2%) 섞여 있었는데
아무도 못 걸렀다. 원인은 수집 단계에 길이 하한 검사가 없었던 것이다(fsd50k_fetcher 가 원본
클립을 길이 검사 없이 그대로 저장). 눈에 띈 계기도 우연이었다 — eval 로그에 test 12개가
"Mel too short" 로 찍혀서였고, 그때는 train 44개가 같은 이유로 통째로 버려지고 있다는 것을
몰랐다. data_provenance.xlsx 에 길이 컬럼조차 없어서 사후 확인도 불가능했다.
=> 수집·분할·증강 어느 단계 뒤에도 돌릴 수 있는 정식 검증 스크립트로 만든다.

검사 항목(2026-07-15 스크리닝에서 실제로 쓰인 기준을 제거된 71개 실측으로 역산해 정식화):
  [CRITICAL] 읽기 실패 / 샘플레이트·채널 불일치 / 길이 < critical_duration(1.0초)
             -> 1.0초 미만은 파서(vild_parser_common.load_and_segment)가 mel 프레임수 <
                segment_length(101) 라서 빈 리스트를 반환한다. 파일이 학습·평가에서 통째로 빠진다.
  [ERROR]    길이 < min_duration(3.0초)
             -> 서로 다른 5개 세그먼트에 101 + 4*50 = 301프레임 = 48,000샘플 = 정확히 3.0초가
                필요하다. 미달이면 마지막 세그먼트를 복제하는 열화 폴백이 걸린다(파서 145~147행).
             클리핑(clip_ratio) / 안들림(rms) / 진폭극소(peak)
  [WARN]     긴 무음구간 / DC 오프셋 / sha256 중복 / provenance-디스크 불일치 / 인덱스 불일치

기준 타당성 검증(2026-07-31):
  - data/_removed_20260715 의 71개(7/15 에 사람이 수작업으로 걸러낸 것)에 돌려 71/71 전부 검출.
  - 같은 기준으로 active 2860개를 검사하면 218개(길이 206 + 3초 이상인데 안 들리는 12개)가 걸린다.
  - 판정에 쓰지 않기로 한 지표는 measure() 아래 주석에 이유를 적어 두었다(오탐 근거 포함).

임계값 근거(data/_removed_20260715 의 71개를 실측해 provenance 사유 문자열과 대조):
  - 클리핑 : 사유 "클리핑0.5%~12.6%" == |x| >= 0.999 인 샘플 비율(실측 일치). 최소 제거값 0.005.
  - 안들림 : 사유 "안들림rms0.0004~0.0018" == 전체 RMS(실측 일치). 최대 제거값 0.0018.
  - 초단타 : 사유 "초단타0.30s~0.49s" == 길이(실측 일치). 당시 기준은 0.5초였고 3.0초로 강화한다.
  - 무음   : 사유 "무음0.99/1.00" 은 정의를 소수점까지는 복원하지 못했다(진폭이 극소한 파일들에
             붙어 있었다). 그 파일들(others_train_117/120/156)은 rms 0.0021~0.0036 으로 안들림
             기준은 통과하지만 peak 가 0.0104~0.0149 로 사실상 들리지 않는다. 그래서 peak 하한을
             별도 지표로 둔다. 긴 무음구간(others_train_421: 연속 27.4초 무음)은 max_silence_run 이 잡는다.

사용 예:
  # 현재 상태 점검만
  python preprocessing/validate_dataset.py --mark_version mark4.8
  # 측정치를 provenance 에 기록(duration_sec 등 컬럼 갱신)
  python preprocessing/validate_dataset.py --mark_version mark4.8 --update_provenance
  # CI/파이프라인용(문제가 있으면 종료코드 1)
  python preprocessing/validate_dataset.py --mark_version mark4.8 --strict
"""
import os
import sys
import csv
import time
import shutil
import hashlib
import argparse

import numpy as np
import pandas as pd
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))            # preprocessing
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))     # mark4_refactored

# 파서가 요구하는 값(vild_config.py 와 일치해야 함). 아래 resolve_parser_limits() 가 실제
# config 를 읽어 덮어쓰므로 여기 값은 config 를 못 읽을 때의 안전 기본값이다.
DEFAULT_SEGMENT_LENGTH = 101
DEFAULT_SEGMENT_HOP = 50
DEFAULT_MAX_SEGMENTS = 5
DEFAULT_HOP_LENGTH = 160
DEFAULT_SR = 16000


def resolve_parser_limits(mark_version):
    """vild_config.py 에서 실제 파서 파라미터를 읽어 (critical_sec, min_sec, sr) 를 계산한다.
    - critical_sec : 이보다 짧으면 load_and_segment 가 [] 반환(파일 통째 제외)
    - min_sec      : 이보다 짧으면 서로 다른 max_segments 개를 못 만들어 복제 폴백(열화)
    config 를 못 읽으면 기본 상수로 계산하고 그 사실을 알린다."""
    vals = {"segment_length": DEFAULT_SEGMENT_LENGTH, "segment_hop": DEFAULT_SEGMENT_HOP,
            "max_segments": DEFAULT_MAX_SEGMENTS, "hop_length": DEFAULT_HOP_LENGTH,
            "sample_rate": DEFAULT_SR}
    src = "기본 상수"
    # [중요] vild_config 를 import 하면 torch 가 딸려온다. 이 venv 는 /mnt/c 에 있어서 torch 로딩에만
    # 약 76초가 걸린다(2026-07-31 실측). 여기서 필요한 건 정수 상수 5개뿐이므로 소스를 텍스트로
    # 읽어 그 값만 뽑는다. 파싱에 실패한 항목은 위 기본 상수를 그대로 쓴다.
    cfg_path = os.path.join(PROJECT_ROOT, "vild", "vild_config.py")
    try:
        import re
        with open(cfg_path, encoding="utf-8") as f:
            text = f.read()
        got = []
        for key in vals:
            m = re.search(rf"^\s*self\.{key}\s*=\s*(\d+)", text, re.MULTILINE)
            if m:
                vals[key] = int(m.group(1))
                got.append(key)
        if got:
            src = f"vild_config.py({len(got)}/{len(vals)}개 파싱)"
    except Exception as e:
        print(f"[WARN] vild_config.py 를 읽지 못해 기본 상수를 씁니다({e})")
    seg_len, seg_hop = vals["segment_length"], vals["segment_hop"]
    max_seg, hop, sr = vals["max_segments"], vals["hop_length"], vals["sample_rate"]

    # mel 프레임 t 개를 만들려면 대략 (t-1)*hop 샘플이 필요하다(center=True 기준).
    critical_sec = (seg_len - 1) * hop / sr
    need_frames = seg_len + (max_seg - 1) * seg_hop
    min_sec = (need_frames - 1) * hop / sr
    print(f"[기준] {src}: segment_length={seg_len}, segment_hop={seg_hop}, "
          f"max_segments={max_seg}, hop_length={hop}, sr={sr}")
    print(f"[기준] 통째 제외 한계 = {critical_sec:.2f}초 미만, "
          f"열화 폴백 한계 = {min_sec:.2f}초 미만({need_frames}프레임 필요)")
    return critical_sec, min_sec, sr


# ===================== 오디오 지표 =====================
def frame_rms(x, sr, frame_sec=0.025, hop_sec=0.010):
    """프레임별 RMS. 파일이 수천 개라 파이썬 루프 대신 sliding window 로 벡터화한다."""
    fr = max(1, int(sr * frame_sec))
    fhop = max(1, int(sr * hop_sec))
    if len(x) < fr:
        return np.array([])
    w = np.lib.stride_tricks.sliding_window_view(x, fr)[::fhop]
    return np.sqrt(np.mean(w.astype(np.float64) ** 2, axis=1) + 1e-12)


def max_run_seconds(mask, sr):
    """mask(True=무음)가 연속으로 이어지는 최대 길이(초). 벡터화 구현."""
    if len(mask) == 0 or not mask.any():
        return 0.0
    d = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return float((ends - starts).max() / sr)


def measure(path, silence_floor, clip_floor):
    """파일 하나의 지표를 잰다. 실패하면 error 에 사유가 담긴다."""
    m = {"path": path, "filename": os.path.basename(path), "error": ""}
    # 파일을 한 번만 연다(sf.read + sf.info 로 두 번 열면 /mnt/c 에서 비용이 두 배가 된다).
    try:
        with sf.SoundFile(path) as f:
            sr, ch, subtype = f.samplerate, f.channels, f.subtype
            x = f.read(dtype="float32", always_2d=True)
    except Exception as e:
        m["error"] = f"read_failed: {e}"
        return m
    x = x.mean(axis=1)
    if len(x) == 0:
        m.update({"error": "empty", "samplerate": sr, "channels": ch})
        return m
    ax = np.abs(x)
    # 활성/무음 판정은 샘플이 아니라 프레임 RMS 로 한다. 정상 오디오도 파형이 0 을 지나가므로
    # 샘플 단위 절대값으로 세면 멀쩡한 소리도 "무음 비율 70%" 처럼 나온다.
    # 정의는 augment_density.active_ratio 와 동일하게 맞춰 둔다(같은 데이터에 두 잣대를 쓰지 않도록).
    fr = frame_rms(x, sr)
    if len(fr):
        thr = max(1e-4, 0.05 * float(fr.max()))
        act = fr > thr
        active_ratio = float(act.mean())
        active_sec = float(act.sum() * 0.010)          # hop 10ms
    else:
        active_ratio, active_sec = 0.0, 0.0
    m.update({
        "samplerate": sr,
        "channels": ch,
        "subtype": subtype,
        "duration_sec": len(x) / sr,
        "rms": float(np.sqrt(np.mean(x ** 2))),
        "peak": float(ax.max()),
        "clip_ratio": float(np.mean(ax >= clip_floor)),
        "silence_ratio": 1.0 - active_ratio,
        # 소리가 실제로 들어있는 총 시간. 파서가 salient_topk 로 소리 있는 창을 골라내므로,
        # 긴 파일 중간의 무음 자체는 문제가 아니다. 문제는 "쓸 소리가 얼마나 있느냐"다.
        "active_sec": active_sec,
        # 무음 패딩 비율(fix_audio_length 가 3초를 채우려고 뒤에 붙인 0 구간을 잡는다)
        "pad_ratio": float(np.mean(ax < 1e-5)),
        "max_silence_sec": max_run_seconds(ax < silence_floor, sr),
        "dc_offset": float(np.mean(x)),
        "size_bytes": os.path.getsize(path),
    })
    return m


def check_waveform(x, sr, min_duration=3.0, min_rms=0.002, min_peak=0.02,
                   max_clip_ratio=0.005, clip_floor=0.999):
    """수집 단계(fsd50k_fetcher / aihub_slicer)에서 쓰는 간이 검사.

    저장하기 전에 파형을 그대로 넘기면, 채택하면 안 되는 이유를 짧은 한국어 문자열로 돌려준다
    (문제가 없으면 None). 검증 스크립트와 같은 임계값을 쓰도록 이 함수를 공유해서, 수집 기준과
    사후 검증 기준이 어긋나는 일이 없게 한다.

    2026-07-31 신설. 이전에는 fsd50k_fetcher 가 원본 클립을 길이 검사 없이 저장해서 0.5초짜리가
    206개 섞여 들어갔고, 그것을 걸러줄 장치가 어디에도 없었다.
    """
    import numpy as _np
    if x is None or len(x) == 0:
        return "빈 파형"
    dur = len(x) / sr
    if dur < min_duration:
        return f"길이부족 {dur:.2f}s(<{min_duration}s)"
    ax = _np.abs(x)
    rms = float(_np.sqrt(_np.mean(_np.asarray(x, dtype=_np.float64) ** 2)))
    if rms < min_rms:
        return f"안들림rms{rms:.4f}"
    peak = float(ax.max())
    if peak < min_peak:
        return f"진폭극소peak{peak:.4f}"
    clip = float(_np.mean(ax >= clip_floor))
    if clip >= max_clip_ratio:
        return f"클리핑{clip*100:.1f}%"
    return None


def reason_text(m):
    """제거 사유를 2026-07-15 provenance 표기와 같은 형식의 짧은 한국어로 만든다.
    예: '초단타0.48s', '클리핑1.6%', '안들림rms0.0004', '진폭극소peak0.0104'"""
    codes = str(m.get("codes", "") or "")
    p = []

    def num(key, default=float("nan")):
        v = m.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    if "READ_FAILED" in codes or "empty" in str(m.get("error", "")):
        p.append("읽기실패")
    if "TOO_SHORT" in codes:
        p.append(f"초단타{num('duration_sec'):.2f}s")
    if "CLIPPING" in codes:
        p.append(f"클리핑{num('clip_ratio')*100:.1f}%")
    if "INAUDIBLE_RMS" in codes:
        p.append(f"안들림rms{num('rms'):.4f}")
    if "INAUDIBLE_PEAK" in codes:
        p.append(f"진폭극소peak{num('peak'):.4f}")
    if "SAMPLERATE" in codes:
        p.append(f"SR{m.get('samplerate')}")
    if "CHANNELS" in codes:
        p.append(f"채널{m.get('channels')}")
    return ";".join(p) if p else (codes or "불명")


def sha256_of(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ===================== 판정 =====================
def judge(m, a, critical_sec, min_sec):
    """지표 dict -> [(severity, code, detail), ...]"""
    out = []
    if m.get("error"):
        return [("CRITICAL", "READ_FAILED", m["error"])]

    d = m["duration_sec"]
    if d < critical_sec:
        out.append(("CRITICAL", "TOO_SHORT_DROPPED",
                    f"{d:.3f}s < {critical_sec:.2f}s — 파서가 파일을 통째로 버립니다(Mel too short)"))
    elif d < min_sec:
        out.append(("ERROR", "TOO_SHORT_DEGRADED",
                    f"{d:.3f}s < {min_sec:.2f}s — 세그먼트를 복제하는 열화 폴백이 걸립니다"))

    if m["samplerate"] != a.expect_sr:
        out.append(("CRITICAL", "SAMPLERATE",
                    f"{m['samplerate']}Hz != {a.expect_sr}Hz"))
    if m["channels"] != a.expect_channels:
        out.append(("CRITICAL", "CHANNELS",
                    f"{m['channels']}ch != {a.expect_channels}ch"))

    if m["rms"] < a.min_rms:
        out.append(("ERROR", "INAUDIBLE_RMS",
                    f"rms {m['rms']:.6f} < {a.min_rms} — 사실상 들리지 않습니다"))
    if m["peak"] < a.min_peak:
        out.append(("ERROR", "INAUDIBLE_PEAK",
                    f"peak {m['peak']:.6f} < {a.min_peak} — 진폭이 극도로 작습니다"))
    if m["clip_ratio"] >= a.max_clip_ratio:
        out.append(("ERROR", "CLIPPING",
                    f"클리핑 {m['clip_ratio']*100:.2f}% >= {a.max_clip_ratio*100:.2f}%"))
    # [주의] 아래 세 지표(silence_ratio / active_sec / pad_ratio)는 측정만 하고 오류 판정에는
    # 쓰지 않는다. 2026-07-31 실측으로 판별력이 없음을 확인했기 때문이다:
    #  - silence_ratio·active_sec 는 상대 임계(0.05*max) 기반이라 "밀도" 지표다. 개 짖는 소리처럼
    #    짧고 강한 이벤트는 원래 값이 낮다(로그 실측: dog 0.40 / others 0.79). 소리 유무 판정에
    #    쓰면 정상 파일 1089개가 걸리는 오탐이 났다.
    #  - pad_ratio 는 무음 패딩을 잡으려 했으나, 임계 0.3~0.8 어디서도 정상 2860개 중 31~362개가
    #    걸리는 동안 실제 불량 71개 중에서는 1~18개만 걸려 판별력이 없었다. 패딩으로 길이만 채운
    #    파일은 duration_sec 을 provenance 에 기록해 두는 쪽이 정확하다(--update_provenance).
    # 긴 무음도 파서가 salient_topk 로 피해가므로 참고 정보(WARN)로만 남긴다.
    if m["max_silence_sec"] >= a.max_silence_run:
        out.append(("WARN", "LONG_SILENCE",
                    f"연속 무음 {m['max_silence_sec']:.2f}s (소리 구간 {m['active_sec']:.2f}s 는 확보됨)"))
    if abs(m["dc_offset"]) >= a.max_dc_offset:
        out.append(("WARN", "DC_OFFSET", f"DC 오프셋 {m['dc_offset']:.5f}"))
    return out


# ===================== 메인 =====================
def main():
    ap = argparse.ArgumentParser(
        description="데이터셋 유효성 검증(길이·음량·클리핑·무음·포맷·중복·정합성)")
    ap.add_argument("--mark_version", type=str, required=True)
    ap.add_argument("--data_root", type=str, default=os.path.join(PROJECT_ROOT, "data"))
    ap.add_argument("--splits", type=str, default="train,val,test")
    ap.add_argument("--provenance_path", type=str,
                    default=os.path.join(PROJECT_ROOT, "..", "data_provenance.xlsx"))
    ap.add_argument("--active_column", type=str, default="removed_20260715",
                    help="provenance 에서 active 판정에 쓰는 컬럼(기존 스크립트와 동일)")
    # 길이 기준: 지정하지 않으면 vild_config 에서 자동 계산
    ap.add_argument("--min_duration", type=float, default=None,
                    help="열화 폴백 한계(기본: config 로 자동계산 = 3.0초)")
    ap.add_argument("--critical_duration", type=float, default=None,
                    help="통째 제외 한계(기본: config 로 자동계산 = 1.0초)")
    # 품질 기준(2026-07-15 실측 역산값)
    ap.add_argument("--min_rms", type=float, default=0.002,
                    help="전체 RMS 하한(2026-07-15 제거분 최대값 0.0018 기준)")
    ap.add_argument("--min_peak", type=float, default=0.02,
                    help="최대진폭 하한. 0.05 는 정상 54개를 잡는 과검출이라 0.02 로 확정"
                         "(제거분 others_train_117/120/156 의 peak 0.0104~0.0149 를 잡는 값)")
    ap.add_argument("--max_clip_ratio", type=float, default=0.005,
                    help="|x|>=clip_floor 인 샘플 비율 상한(제거분 최소값 0.005 기준)")
    ap.add_argument("--clip_floor", type=float, default=0.999)
    ap.add_argument("--silence_floor", type=float, default=0.01,
                    help="max_silence_sec 계산용 무음 판정 진폭")
    ap.add_argument("--max_silence_run", type=float, default=3.0,
                    help="연속 무음이 이보다 길면 WARN(파서가 salient_topk 로 피해가므로 오류는 아님)")
    ap.add_argument("--max_dc_offset", type=float, default=0.01)
    ap.add_argument("--expect_sr", type=int, default=16000)
    ap.add_argument("--expect_channels", type=int, default=1)
    # 동작
    ap.add_argument("--skip_hash", action="store_true", help="sha256 중복검사 생략(빠름)")
    ap.add_argument("--skip_provenance", action="store_true",
                    help="provenance 대조 생략. 엑셀 2931행 읽기에만 약 19초가 든다(파일 검사만 할 때 유용)")
    ap.add_argument("--report_csv", type=str, default=None)
    ap.add_argument("--update_provenance", action="store_true",
                    help="측정치를 provenance 에 기록(duration_sec 등). 실행 전 백업 생성")
    ap.add_argument("--quarantine", action="store_true",
                    help="CRITICAL/ERROR 파일을 data/_removed_<오늘날짜>/<split>/ 로 옮기고 "
                         "provenance 의 active 컬럼에 제거 사유를 적는다(2026-07-15 방식과 동일)")
    ap.add_argument("--check_index", action="store_true",
                    help="preprocessing/dataset_index_<mark>.csv 와 대조")
    ap.add_argument("--strict", action="store_true", help="CRITICAL/ERROR 가 있으면 종료코드 1")
    a = ap.parse_args()

    critical_sec, min_sec, cfg_sr = resolve_parser_limits(a.mark_version)
    if a.critical_duration is not None:
        critical_sec = a.critical_duration
    if a.min_duration is not None:
        min_sec = a.min_duration
    if a.expect_sr != cfg_sr:
        print(f"[WARN] --expect_sr({a.expect_sr}) 가 config sample_rate({cfg_sr}) 와 다릅니다")

    splits = [s.strip() for s in a.splits.split(",") if s.strip()]
    print(f"\n[검증] mark_version={a.mark_version}  data_root={a.data_root}  splits={splits}")

    # ---------- 1) 파일 수집 + 측정 ----------
    files = []
    for sp in splits:
        d = os.path.join(a.data_root, sp)
        if not os.path.isdir(d):
            print(f"[WARN] 폴더 없음: {d}")
            continue
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(".wav"):
                files.append((sp, os.path.join(d, fn)))
    if not files:
        print("[ERROR] 검사할 wav 가 없습니다.")
        sys.exit(1)

    print(f"[검증] 파일 {len(files)}개 측정 중...")
    t0 = time.time()
    records, findings = [], []
    for i, (sp, path) in enumerate(files, 1):
        m = measure(path, a.silence_floor, a.clip_floor)
        m["split"] = sp
        m["label"] = "dog_bark" if os.path.basename(path).startswith("dog_bark") else "others"
        if not a.skip_hash and not m.get("error"):
            m["sha256"] = sha256_of(path)
        issues = judge(m, a, critical_sec, min_sec)
        m["severity"] = ("CRITICAL" if any(s == "CRITICAL" for s, _, _ in issues)
                         else "ERROR" if any(s == "ERROR" for s, _, _ in issues)
                         else "WARN" if issues else "OK")
        m["codes"] = ";".join(c for _, c, _ in issues)
        m["details"] = " | ".join(d for _, _, d in issues)
        records.append(m)
        for sev, code, det in issues:
            findings.append({"split": sp, "filename": m["filename"], "severity": sev,
                             "code": code, "detail": det})
        if i % 500 == 0 or i == len(files):
            print(f"  {i}/{len(files)} ({time.time()-t0:.0f}초)")

    df = pd.DataFrame(records)

    # ---------- 2) 중복 ----------
    dup_rows = []
    if not a.skip_hash and "sha256" in df.columns:
        ok = df[df["sha256"].notna()]
        dups = ok[ok.duplicated("sha256", keep=False)].sort_values("sha256")
        for h, g in dups.groupby("sha256"):
            names = list(g["filename"])
            dup_rows.append({"sha256": h, "count": len(names), "files": ", ".join(names)})
            for n in names:
                findings.append({"split": "-", "filename": n, "severity": "WARN",
                                 "code": "DUPLICATE", "detail": f"동일 내용 {len(names)}개: {', '.join(names)}"})

    # ---------- 3) provenance 정합성 ----------
    prov_path = os.path.abspath(a.provenance_path)
    prov = None
    only_disk = only_prov = set()
    if a.skip_provenance:
        print("[건너뜀] provenance 대조(--skip_provenance)")
    elif os.path.exists(prov_path):
        prov = pd.read_excel(prov_path)
        # [변경 2026-08-01] provenance 는 mark4.x 전 버전이 한 파일을 공유한다. 디스크와 대조할 때는
        # 이 버전의 행만 봐야 한다. 안 그러면 mark4.6 을 검증할 때 mark4.8 행 2,860개가 전부
        # MISSING_FILE 로 잡혀서 진짜 문제가 그 안에 묻힌다.
        # prov 자체는 아래 --update_provenance / --quarantine 이 파일 전체를 다시 쓰는 데 쓰이므로
        # 원본 그대로 두고, 걸러낸 것은 scoped 로 따로 둔다.
        scoped = prov
        if "mark_version" in prov.columns:
            scoped = prov[prov["mark_version"] == a.mark_version]
            print(f"[INFO] provenance 전체 {len(prov)}행 중 {a.mark_version} {len(scoped)}행으로 대조")
        if a.active_column in scoped.columns:
            act = scoped[scoped[a.active_column] == "active"]
        else:
            print(f"[WARN] provenance 에 '{a.active_column}' 컬럼이 없어 전체를 active 로 봅니다")
            act = scoped
        pv = set(act["local_filename"].astype(str))
        dk = set(df["filename"])
        only_disk, only_prov = dk - pv, pv - dk
        for n in sorted(only_disk):
            findings.append({"split": "-", "filename": n, "severity": "WARN",
                             "code": "ORPHAN_FILE", "detail": "디스크에는 있는데 provenance active 에 없음"})
        for n in sorted(only_prov):
            findings.append({"split": "-", "filename": n, "severity": "WARN",
                             "code": "MISSING_FILE", "detail": "provenance active 인데 디스크에 없음"})
    else:
        print(f"[WARN] provenance 없음: {prov_path}")
        only_disk = only_prov = set()

    # ---------- 4) 인덱스 정합성 ----------
    idx_only = disk_only = set()
    if a.check_index:
        idx_path = os.path.join(BASE_DIR, f"dataset_index_{a.mark_version}.csv")
        if os.path.exists(idx_path):
            idx = pd.read_csv(idx_path)
            names = {os.path.basename(p) for p in idx["path"].astype(str)}
            idx_only, disk_only = names - set(df["filename"]), set(df["filename"]) - names
            for n in sorted(idx_only):
                findings.append({"split": "-", "filename": n, "severity": "WARN",
                                 "code": "INDEX_ONLY", "detail": "인덱스에 있는데 디스크에 없음"})
            for n in sorted(disk_only):
                findings.append({"split": "-", "filename": n, "severity": "WARN",
                                 "code": "INDEX_MISSING", "detail": "디스크에 있는데 인덱스에 없음"})
        else:
            print(f"[WARN] 인덱스 없음: {idx_path}")

    # ---------- 5) 리포트 ----------
    fdf = pd.DataFrame(findings)
    print("\n" + "=" * 72)
    print(f"검증 결과 — 총 {len(df)}개 파일")
    print("=" * 72)
    good = df[df["error"] == ""] if "error" in df.columns else df
    if len(good):
        print(f"길이(초): min {good['duration_sec'].min():.3f} / "
              f"median {good['duration_sec'].median():.3f} / max {good['duration_sec'].max():.3f}")
    n_crit = int((df["severity"] == "CRITICAL").sum())
    n_err = int((df["severity"] == "ERROR").sum())
    n_ok = int((df["severity"] == "OK").sum())
    print(f"판정: OK {n_ok} / WARN {int((df['severity']=='WARN').sum())} / "
          f"ERROR {n_err} / CRITICAL {n_crit}")

    if len(fdf):
        print("\n[코드별 건수]")
        for code, g in fdf.groupby("code"):
            sev = g["severity"].iloc[0]
            print(f"  {sev:<8} {code:<20} {len(g):>5}건")
        print("\n[split × 코드]")
        piv = fdf[fdf["split"] != "-"].pivot_table(index="split", columns="code",
                                                   values="filename", aggfunc="count", fill_value=0)
        if len(piv):
            print(piv.to_string())
        print("\n[상위 20건]")
        order = {"CRITICAL": 0, "ERROR": 1, "WARN": 2}
        for _, r in fdf.assign(_o=fdf["severity"].map(order)).sort_values(
                ["_o", "code", "filename"]).head(20).iterrows():
            print(f"  [{r['severity']}] {r['filename']:<28} {r['code']:<20} {r['detail']}")
    else:
        print("\n문제 없음 — 모든 검사를 통과했습니다.")

    if dup_rows:
        print(f"\n[중복] 동일 sha256 그룹 {len(dup_rows)}개")
        for r in dup_rows[:10]:
            print(f"  {r['count']}개: {r['files']}")
    if prov is not None:
        print(f"\n[정합성] 디스크 {len(df)} / provenance active {len(pv)} — "
              f"디스크에만 {len(only_disk)}, provenance에만 {len(only_prov)}")

    # CSV 저장
    out = a.report_csv
    if out is None:
        log_dir = os.path.join(PROJECT_ROOT, "logFiles")
        os.makedirs(log_dir, exist_ok=True)
        out = os.path.join(log_dir,
                           f"validation_{a.mark_version}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    cols = ["split", "label", "filename", "severity", "codes", "duration_sec", "active_sec",
            "pad_ratio", "rms", "peak", "clip_ratio", "silence_ratio", "max_silence_sec",
            "dc_offset", "samplerate", "channels", "subtype", "size_bytes", "details"]
    df.reindex(columns=[c for c in cols if c in df.columns]).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[저장] 상세 리포트: {out}")

    # ---------- 6) provenance 갱신 ----------
    if a.update_provenance and prov is not None:
        bak = prov_path + f".bak_before_validate_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(prov_path, bak)
        meas = df.set_index("filename")
        # [추가 2026-08-01] 파일명만 보고 행을 찾으면 다른 버전의 같은 이름 행에 측정치가 덮인다
        # (others_train_001.wav 는 모든 mark 버전에 생긴다). 이 버전의 행에만 쓴다.
        ismine = ((prov["mark_version"] == a.mark_version) if "mark_version" in prov.columns
                  else pd.Series(True, index=prov.index))
        for col, src in [("duration_sec", "duration_sec"), ("active_sec", "active_sec"),
                         ("pad_ratio", "pad_ratio"), ("rms", "rms"), ("peak", "peak"),
                         ("clip_ratio", "clip_ratio"), ("silence_ratio", "silence_ratio"),
                         ("max_silence_sec", "max_silence_sec")]:
            if col not in prov.columns:
                prov[col] = np.nan
            mapped = prov["local_filename"].astype(str).map(
                meas[src] if src in meas.columns else {})
            prov[col] = mapped.where(ismine).combine_first(prov[col])
        if "validated_date" not in prov.columns:
            prov["validated_date"] = pd.NA
        hit = prov["local_filename"].astype(str).isin(set(df["filename"])) & ismine
        prov.loc[hit, "validated_date"] = time.strftime("%Y-%m-%d")
        prov.to_excel(prov_path, index=False)
        print(f"[저장] provenance 갱신({int(hit.sum())}행에 측정치 기록) — 백업 {os.path.basename(bak)}")

    # ---------- 7) 격리 ----------
    if a.quarantine:
        bad = df[df["severity"].isin(["CRITICAL", "ERROR"])]
        if bad.empty:
            print("\n[격리] 옮길 파일이 없습니다.")
        else:
            stamp = time.strftime("%Y%m%d")
            qroot = os.path.join(a.data_root, f"_removed_{stamp}")
            moved, reasons = [], {}
            for _, r in bad.iterrows():
                reasons[r["filename"]] = "제거: " + reason_text(r)
            for _, r in bad.iterrows():
                dst_dir = os.path.join(qroot, r["split"])
                os.makedirs(dst_dir, exist_ok=True)
                src = r["path"]
                dst = os.path.join(dst_dir, r["filename"])
                if os.path.exists(src):
                    shutil.move(src, dst)
                    moved.append(r["filename"])
            print(f"\n[격리] {len(moved)}개를 {qroot} 로 옮겼습니다.")
            if prov is not None and moved:
                bak = prov_path + f".bak_before_quarantine_{time.strftime('%Y%m%d_%H%M%S')}"
                shutil.copy2(prov_path, bak)
                col = a.active_column
                if col not in prov.columns:
                    prov[col] = "active"
                hit = prov["local_filename"].astype(str).isin(set(moved)) & (prov[col] == "active")
                # [추가 2026-08-01] 같은 파일명이 다른 버전에도 있을 수 있으므로 버전 조건을 건다.
                # 이게 없으면 4.6 에서 격리한 파일 때문에 4.8 의 멀쩡한 행이 '제거'로 표시된다.
                if "mark_version" in prov.columns:
                    hit &= (prov["mark_version"] == a.mark_version)
                prov.loc[hit, col] = prov.loc[hit, "local_filename"].astype(str).map(reasons)
                prov.to_excel(prov_path, index=False)
                print(f"[격리] provenance {int(hit.sum())}행에 사유 기록 — 백업 {os.path.basename(bak)}")
                from collections import Counter
                for rs, c in Counter(reasons.values()).most_common(8):
                    print(f"    {rs} : {c}개")

    if a.strict and (n_crit or n_err):
        print(f"\n[STRICT] CRITICAL {n_crit} / ERROR {n_err} — 종료코드 1")
        sys.exit(1)
    print("\n[완료]")


if __name__ == "__main__":
    main()
