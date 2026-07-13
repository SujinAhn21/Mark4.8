"""
fsd50k_fetcher.py — FSD50K에서 필요한 클립만 골라 받는 수집 스크립트(4.x 시리즈 공용).

배경(2026-07-13): mark4.8 성능 개선 가설5(실데이터 증량)를 위해, FSD50K 전체(약 24GB)를
받지 않고 Zenodo 분할 zip(multi-volume archive)에서 HTTP Range 요청으로 개별 wav만
선택 추출한다(central directory 파싱 -> local file header 파싱 -> zlib 해제 -> CRC32 검증).
2026-07-11 최초 수집(클래스당 200개) 때 쓴 기법을 재사용 가능한 정식 스크립트로 만든 것.

동작:
1. ground truth CSV(dev/eval)를 내려받아(캐시) 라벨 기준으로 후보를 고른다.
   - target 클래스: --target_labels 의 라벨을 전부 가진 클립 (mark4.8: Bark,Dog)
   - others: 차량/말소리/생활소음 3대 기둥에서 기존 수집 구성비(약 45/35/20)로 샘플링.
     Dog/Bark 계열 라벨이 하나라도 있으면 제외.
2. data_provenance.xlsx 의 fsd50k_fname 과 대조해 이미 쓴 클립은 건너뛴다(중복 방지).
3. 선택된 클립을 Range 요청으로 추출 -> 모노 -> 16kHz 리샘플 -> float32 wav 로
   data/{split}/{class}_{split}_{NNN}.wav 에 저장(기존 번호 이어서).
   3초 미만 패딩은 여기서 하지 않는다(run_all step0 의 fix_audio_length 가 처리).
4. data_provenance.xlsx 에 행 추가(실행 전 백업 생성, 20개마다 중간 저장).

사용 예:
  python preprocessing/fsd50k_fetcher.py --mark_version mark4.8 --dry_run
  python preprocessing/fsd50k_fetcher.py --mark_version mark4.8 --target_class dog_bark \
      --target_labels Bark,Dog
"""
import os
import io
import sys
import csv
import time
import zlib
import struct
import shutil
import hashlib
import zipfile
import argparse
from datetime import date

import numpy as np
import pandas as pd
import requests
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # preprocessing
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

ZENODO_RECORD_API = "https://zenodo.org/api/records/4060432"
ZENODO_FILE_URL = "https://zenodo.org/records/4060432/files/{name}?download=1"

# others 3대 기둥(기존 수집 구성: 차량/교통 ~50%, 말소리 ~36%, 생활소음 ~30%; 겹침 있음)
PILLAR_VEHICLE = {
    "Vehicle", "Motor_vehicle_(road)", "Car", "Engine", "Truck", "Bus",
    "Motorcycle", "Traffic_noise_and_roadway_noise", "Rail_transport", "Train",
}
PILLAR_SPEECH = {
    "Human_voice", "Speech", "Male_speech_and_man_speaking",
    "Female_speech_and_woman_speaking", "Conversation", "Child_speech_and_kid_speaking",
}
PILLAR_DOMESTIC = {
    "Domestic_sounds_and_home_sounds", "Door", "Alarm", "Sliding_door", "Doorbell",
    "Microwave_oven", "Water_tap_and_faucet", "Dishes_and_pots_and_pans",
    "Cutlery_and_silverware", "Telephone",
}
# others 에서 제외할 라벨(타겟 오염 방지). --exclude_labels 로 덮어쓸 수 있음.
DEFAULT_EXCLUDE = {"Dog", "Bark", "Growling", "Howl", "Whimper_(dog)", "Bow-wow"}
# 기둥별 샘플링 비율(vehicle, speech, domestic) — 기존 200개 구성비 근사
PILLAR_RATIO = (0.45, 0.35, 0.20)


# ===================== 원격 분할 zip 리더 =====================
class RemoteSplitZip:
    """Zenodo 의 분할 zip(.z01..zNN + .zip)에서 개별 파일을 Range 요청으로 추출한다.
    분할 zip 은 논리적으로 하나의 스트림이 볼륨 여러 개로 잘린 것이고, central directory 의
    각 엔트리는 (시작 디스크 번호, 그 디스크 안에서의 local header 오프셋)을 기록한다."""

    def __init__(self, volumes, session=None):
        # volumes: 디스크 순서(z01..zNN, 마지막이 .zip)의 [(url, size), ...]
        self.volumes = volumes
        self.session = session or requests.Session()
        self.entries = {}   # basename(확장자 제외) -> dict
        self._load_central_directory()

    # ---- 저수준 읽기 ----
    def _read_range(self, url, start, length, max_retry=5):
        end = start + length - 1
        for attempt in range(max_retry):
            try:
                r = self.session.get(
                    url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
                if r.status_code in (200, 206) and len(r.content) >= length:
                    return r.content[:length]
                raise IOError(f"HTTP {r.status_code}, got {len(r.content)}/{length}B")
            except Exception as e:
                if attempt == max_retry - 1:
                    raise
                wait = 2 ** attempt
                print(f"[WARN] Range 요청 실패({e}) — {wait}초 후 재시도 {attempt+1}/{max_retry}")
                time.sleep(wait)

    def _normalize(self, disk, offset):
        # offset 이 해당 볼륨 크기를 넘으면 다음 볼륨으로 이월
        while offset >= self.volumes[disk][1]:
            offset -= self.volumes[disk][1]
            disk += 1
        return disk, offset

    def read_at(self, disk, offset, length):
        """(disk, offset)에서 length 바이트. 볼륨 경계를 넘으면 다음 볼륨에서 이어 읽는다."""
        disk, offset = self._normalize(disk, offset)
        out = b""
        while length > 0:
            url, size = self.volumes[disk]
            take = min(length, size - offset)
            out += self._read_range(url, offset, take)
            length -= take
            disk += 1
            offset = 0
        return out

    # ---- central directory 파싱 ----
    def _load_central_directory(self):
        last_url, last_size = self.volumes[-1]
        tail_len = min(66000, last_size)
        tail = self._read_range(last_url, last_size - tail_len, tail_len)

        eocd_pos = tail.rfind(b"PK\x05\x06")
        if eocd_pos < 0:
            raise IOError("EOCD 를 찾지 못했습니다(zip 형식 아님?).")
        (_, _, _, n_total, cd_size, cd_offset, _) = struct.unpack(
            "<4H2LH", tail[eocd_pos + 4: eocd_pos + 22])
        cd_disk = struct.unpack("<H", tail[eocd_pos + 6: eocd_pos + 8])[0]

        # Zip64 (개수/크기/오프셋이 최대값이면 EOCD64 에 실제 값이 있음)
        if 0xFFFF in (n_total, cd_disk) or 0xFFFFFFFF in (cd_size, cd_offset):
            loc_pos = tail.rfind(b"PK\x06\x07", 0, eocd_pos)
            if loc_pos < 0:
                raise IOError("Zip64 EOCD locator 를 찾지 못했습니다.")
            eocd64_disk, eocd64_off, _ = struct.unpack(
                "<LQL", tail[loc_pos + 4: loc_pos + 20])
            eocd64 = self.read_at(eocd64_disk, eocd64_off, 56)
            if eocd64[:4] != b"PK\x06\x06":
                raise IOError("Zip64 EOCD 서명 불일치.")
            (_, _, _, _, cd_disk, _, n_total, cd_size, cd_offset) = struct.unpack(
                "<QHHLL4Q", eocd64[4:56])

        cd = self.read_at(cd_disk, cd_offset, cd_size)
        pos, count = 0, 0
        while pos + 46 <= len(cd) and cd[pos:pos + 4] == b"PK\x01\x02":
            (method, crc, csize, usize, nlen, elen, clen, disk_start, loc_off) = (
                struct.unpack("<H", cd[pos + 10:pos + 12])[0],
                struct.unpack("<L", cd[pos + 16:pos + 20])[0],
                struct.unpack("<L", cd[pos + 20:pos + 24])[0],
                struct.unpack("<L", cd[pos + 24:pos + 28])[0],
                struct.unpack("<H", cd[pos + 28:pos + 30])[0],
                struct.unpack("<H", cd[pos + 30:pos + 32])[0],
                struct.unpack("<H", cd[pos + 32:pos + 34])[0],
                struct.unpack("<H", cd[pos + 34:pos + 36])[0],
                struct.unpack("<L", cd[pos + 42:pos + 46])[0],
            )
            name = cd[pos + 46: pos + 46 + nlen].decode("utf-8", "replace")
            extra = cd[pos + 46 + nlen: pos + 46 + nlen + elen]

            # Zip64 extra field(id 0x0001): 최대값(0xFFFF/0xFFFFFFFF)인 필드만 순서대로 들어있음
            ep = 0
            while ep + 4 <= len(extra):
                fid, flen = struct.unpack("<HH", extra[ep:ep + 4])
                if fid == 0x0001:
                    fp = ep + 4
                    if usize == 0xFFFFFFFF:
                        usize = struct.unpack("<Q", extra[fp:fp + 8])[0]; fp += 8
                    if csize == 0xFFFFFFFF:
                        csize = struct.unpack("<Q", extra[fp:fp + 8])[0]; fp += 8
                    if loc_off == 0xFFFFFFFF:
                        loc_off = struct.unpack("<Q", extra[fp:fp + 8])[0]; fp += 8
                    if disk_start == 0xFFFF:
                        disk_start = struct.unpack("<L", extra[fp:fp + 4])[0]; fp += 4
                ep += 4 + flen

            key = os.path.splitext(os.path.basename(name))[0]
            self.entries[key] = {
                "name": name, "method": method, "crc": crc,
                "csize": csize, "usize": usize,
                "disk": disk_start, "offset": loc_off,
            }
            pos += 46 + nlen + elen + clen
            count += 1
        print(f"[INFO] central directory 파싱 완료: {count}개 엔트리")

    # ---- 개별 파일 추출 ----
    def extract(self, key):
        e = self.entries[key]
        header = self.read_at(e["disk"], e["offset"], 30)
        if header[:4] != b"PK\x03\x04":
            raise IOError(f"local header 서명 불일치: {e['name']}")
        nlen, elen = struct.unpack("<HH", header[26:30])
        data = self.read_at(e["disk"], e["offset"] + 30 + nlen + elen, e["csize"])
        if e["method"] == 0:
            raw = data
        elif e["method"] == 8:
            raw = zlib.decompressobj(-15).decompress(data)
        else:
            raise IOError(f"지원하지 않는 압축 방식 {e['method']}: {e['name']}")
        if (zlib.crc32(raw) & 0xFFFFFFFF) != e["crc"]:
            raise IOError(f"CRC32 불일치: {e['name']}")
        return raw


# ===================== 유틸 =====================
def get_zenodo_volumes(session):
    """Zenodo API 에서 파일명/크기를 받아 dev/eval 볼륨 목록(디스크 순서)을 만든다."""
    r = session.get(ZENODO_RECORD_API, timeout=60)
    r.raise_for_status()
    files = {f["key"]: f["size"] for f in r.json()["files"]}
    vols = {}
    for vol in ("dev", "eval"):
        parts = sorted(k for k in files if k.startswith(f"FSD50K.{vol}_audio.z")
                       and not k.endswith(".zip"))
        ordered = parts + [f"FSD50K.{vol}_audio.zip"]   # .zip 이 마지막 디스크
        vols[vol] = [(ZENODO_FILE_URL.format(name=n), files[n]) for n in ordered]
    return vols


def load_ground_truth(cache_dir, session):
    """ground truth CSV 를 내려받아(캐시) {fname:int -> (labels:set, volume:str)} 로 반환."""
    os.makedirs(cache_dir, exist_ok=True)
    zip_path = os.path.join(cache_dir, "FSD50K.ground_truth.zip")
    if not os.path.exists(zip_path):
        print("[INFO] ground truth 다운로드 중...")
        r = session.get(ZENODO_FILE_URL.format(name="FSD50K.ground_truth.zip"), timeout=120)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(r.content)
    gt = {}
    with zipfile.ZipFile(zip_path) as z:
        for vol, member in (("dev", "FSD50K.ground_truth/dev.csv"),
                            ("eval", "FSD50K.ground_truth/eval.csv")):
            with z.open(member) as f:
                for row in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
                    gt[int(row["fname"])] = (set(row["labels"].split(",")), vol)
    print(f"[INFO] ground truth 로드: {len(gt)}개 클립")
    return gt


def to_mono_16k_float32(wav_bytes):
    """zip 에서 꺼낸 wav bytes -> 모노 float32 16kHz numpy 배열."""
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    x = data.mean(axis=1)
    if sr != 16000:
        import torch
        import torchaudio.functional as AF
        t = torch.from_numpy(x).unsqueeze(0)
        x = AF.resample(t, sr, 16000).squeeze(0).numpy()
    return x.astype(np.float32)


def next_index(data_split_dir, prefix):
    """기존 파일 번호에 이어붙일 다음 번호(증강본 _aug_ 는 별도 체계라 제외)."""
    mx = 0
    for p in os.listdir(data_split_dir) if os.path.isdir(data_split_dir) else []:
        if p.startswith(prefix) and p.endswith(".wav") and "_aug_" not in p:
            try:
                mx = max(mx, int(os.path.splitext(p)[0].rsplit("_", 1)[1]))
            except ValueError:
                pass
    return mx + 1


# ===================== 메인 =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mark_version", type=str, required=True)
    ap.add_argument("--target_class", type=str, default="dog_bark")
    ap.add_argument("--target_labels", type=str, default="Bark,Dog",
                    help="타겟 클립이 전부 가져야 하는 FSD50K 라벨(콤마 구분)")
    ap.add_argument("--exclude_labels", type=str, default=",".join(sorted(DEFAULT_EXCLUDE)),
                    help="others 에서 하나라도 있으면 제외할 라벨(콤마 구분)")
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    ap.add_argument("--max_new_per_class", type=int, default=None,
                    help="클래스당 신규 수집 상한(기본: 타겟 가용량 전부)")
    ap.add_argument("--provenance_path", type=str,
                    default=os.path.join(PROJECT_ROOT, "..", "data_provenance.xlsx"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    session = requests.Session()
    rng = np.random.default_rng(args.seed)
    prov_path = os.path.abspath(args.provenance_path)
    data_dir = os.path.join(PROJECT_ROOT, "data", args.split)

    # 1) 기존 사용 클립 파악(중복 방지) + 고아 파일 점검
    prov = pd.read_excel(prov_path)
    used = set(prov.loc[prov["fsd50k_fname"].notna(), "fsd50k_fname"].astype(int))
    known_files = set(prov["local_filename"].astype(str))
    orphans = [p for p in (os.listdir(data_dir) if os.path.isdir(data_dir) else [])
               if p.endswith(".wav") and p not in known_files]
    if orphans:
        print(f"[ERROR] provenance 에 없는 wav 가 {len(orphans)}개 있습니다(이전 실행이 중간에 끊긴 흔적):")
        for p in orphans[:10]:
            print("  -", p)
        print("이 파일들을 정리(삭제 또는 provenance 반영)한 뒤 다시 실행하십시오.")
        sys.exit(1)
    print(f"[INFO] 기존 사용 클립 {len(used)}개 (provenance {len(prov)}행)")

    # 2) 후보 선정
    gt = load_ground_truth(os.path.join(PROJECT_ROOT, "data", "_fsd50k_meta"), session)
    target_labels = set(args.target_labels.split(","))
    exclude_labels = set(args.exclude_labels.split(","))

    target_cand = [(f, labs, vol) for f, (labs, vol) in gt.items()
                   if target_labels <= labs and f not in used]
    if args.max_new_per_class is not None and len(target_cand) > args.max_new_per_class:
        idx = rng.choice(len(target_cand), size=args.max_new_per_class, replace=False)
        target_cand = [target_cand[i] for i in sorted(idx)]
    n_new = len(target_cand)

    pillars = {"vehicle": [], "speech": [], "domestic": []}
    for f, (labs, vol) in gt.items():
        if f in used or (labs & exclude_labels):
            continue
        if labs & PILLAR_VEHICLE:
            pillars["vehicle"].append((f, labs, vol))
        elif labs & PILLAR_SPEECH:
            pillars["speech"].append((f, labs, vol))
        elif labs & PILLAR_DOMESTIC:
            pillars["domestic"].append((f, labs, vol))
    quota = {"vehicle": round(n_new * PILLAR_RATIO[0]),
             "speech": round(n_new * PILLAR_RATIO[1])}
    quota["domestic"] = n_new - quota["vehicle"] - quota["speech"]
    others_cand = []
    for k in ("vehicle", "speech", "domestic"):
        pool = pillars[k]
        take = min(quota[k], len(pool))
        idx = rng.choice(len(pool), size=take, replace=False)
        others_cand += [pool[i] for i in sorted(idx)]

    print(f"[계획] {args.target_class}: 신규 {n_new}개 "
          f"(dev {sum(1 for c in target_cand if c[2]=='dev')}/"
          f"eval {sum(1 for c in target_cand if c[2]=='eval')})")
    print(f"[계획] others: 신규 {len(others_cand)}개 "
          f"(vehicle {quota['vehicle']}/speech {quota['speech']}/domestic {quota['domestic']})")

    if args.dry_run:
        print("[DRY RUN] 다운로드는 하지 않고 종료합니다.")
        return

    # 3) 원격 zip 리더 준비(필요한 볼륨만)
    volumes = get_zenodo_volumes(session)
    readers = {}
    need_vols = {c[2] for c in target_cand} | {c[2] for c in others_cand}
    for vol in sorted(need_vols):
        print(f"[INFO] {vol} 아카이브 central directory 로드 중...")
        readers[vol] = RemoteSplitZip(volumes[vol], session)

    # 4) 다운로드 + 저장 + provenance 행 추가
    backup = prov_path + f".bak_before_fetch_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(prov_path, backup)
    print(f"[INFO] provenance 백업: {os.path.basename(backup)}")
    os.makedirs(data_dir, exist_ok=True)

    new_rows, failed = [], []
    jobs = ([(args.target_class, c) for c in target_cand]
            + [("others", c) for c in others_cand])
    counters = {args.target_class: next_index(data_dir, f"{args.target_class}_{args.split}_"),
                "others": next_index(data_dir, f"others_{args.split}_")}
    t0 = time.time()
    for i, (cls, (fname, labs, vol)) in enumerate(jobs, 1):
        key = str(fname)
        try:
            raw = readers[vol].extract(key)
            x = to_mono_16k_float32(raw)
            local_name = f"{cls}_{args.split}_{counters[cls]:03d}.wav"
            out_path = os.path.join(data_dir, local_name)
            sf.write(out_path, x, 16000, subtype="FLOAT")
            counters[cls] += 1
            new_rows.append({
                "local_filename": local_name, "fsd50k_fname": fname,
                "fsd50k_split": vol, "original_labels": ",".join(sorted(labs)),
                "target_class": cls, "assigned_split": args.split,
                "mark_version": args.mark_version,
                "sha256": hashlib.sha256(open(out_path, "rb").read()).hexdigest(),
                "source_volume": vol, "size_bytes": os.path.getsize(out_path),
                "download_date": date.today().isoformat(), "source_type": "original",
            })
        except Exception as e:
            failed.append((fname, cls, str(e)))
            print(f"[WARN] 실패 {fname}({cls}): {e}")
        if i % 20 == 0 or i == len(jobs):
            merged = pd.concat([prov, pd.DataFrame(new_rows)], ignore_index=True)
            merged.to_excel(prov_path, index=False)
            el = time.time() - t0
            print(f"[진행] {i}/{len(jobs)} (실패 {len(failed)}) — "
                  f"{el/60:.1f}분 경과, provenance 중간 저장")
        time.sleep(0.15)   # Zenodo 부하/rate limit 배려

    print(f"\n[완료] 신규 저장 {len(new_rows)}개, 실패 {len(failed)}개")
    if failed:
        print("실패 목록(재실행하면 이어서 시도됨 — 이미 받은 것은 provenance 로 중복 방지):")
        for f, c, e in failed[:20]:
            print(f"  - {f} ({c}): {e}")
    print(f"provenance: {prov_path} ({len(prov) + len(new_rows)}행)")
    print("다음 단계: generate_dataset_index.py 재실행 -> augment_density.py 재적용")


if __name__ == "__main__":
    main()
