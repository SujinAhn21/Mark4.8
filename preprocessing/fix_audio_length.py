# fix_audio_length.py 
# 인자 반영

import os
import sys
import torch
import torchaudio
import argparse
from tqdm import tqdm

# === 인자 parsing: default 값을 None으로 변경하여 인자 전달 여부 확인 ===
# [변경 2026-07-11] 5초(80,000샘플) -> 15초(240,000샘플). mark5의 fix_audio_length_to_240000.py와
# 목표 길이를 통일(기존엔 mark4=5초/mark5=15초로 서로 달라 teacher/student 전처리 전제가 어긋나 있었음).
parser = argparse.ArgumentParser(description="오디오 파일 길이를 15초(240000 샘플)로 고정합니다.") # 전처리.
parser.add_argument("--mark_version", type=str, default=None, 
                    help="모델 버전 (예: mark4.1). 이 버전에 따라 입/출력 폴더가 결정됩니다.")
args = parser.parse_args()

# === 기본 경로 설정 ====
# 현재 스크립트가 있는 폴더가 기준이 됨 ( /content/MyProject/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === 보정 파라미터 ===
TARGET_SAMPLE_RATE = 16000 # sample rate 도 제한을 해서 주파수 해상도를 일정하게 맞추기
TARGET_NUM_SAMPLES = TARGET_SAMPLE_RATE * 15  # 240,000 samples (15초 기준, mark5와 통일)

def fix_wav_length(wav_path, save_path):
    try:
        waveform, sr = torchaudio.load(wav_path)

        # 샘플레이트 맞추기(보정하기)
        if sr != TARGET_SAMPLE_RATE:
            resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)
            waveform = resample(waveform)

        # 자르기 or 패딩
        num_samples = waveform.shape[1]
        if num_samples > TARGET_NUM_SAMPLES:
            fixed_waveform = waveform[:, :TARGET_NUM_SAMPLES]
        elif num_samples < TARGET_NUM_SAMPLES:
            pad_len = TARGET_NUM_SAMPLES - num_samples
            pad_tensor = torch.zeros((waveform.shape[0], pad_len))
            fixed_waveform = torch.cat([waveform, pad_tensor], dim=1)
        else:
            fixed_waveform = waveform

        torchaudio.save(save_path, fixed_waveform, TARGET_SAMPLE_RATE)
        # 성공 시 True 반환
        return True
    except Exception as e:
        print(f"\n[ERROR] 파일 처리 중 오류 발생 {os.path.basename(wav_path)}: {e}")
        # 실패 시 False 반환
        return False


def process_all(mark_version):
    # mark_version이 제공되지 않으면 에러 발생
    if mark_version is None:
        print("[CRITICAL ERROR] --mark_version 인자가 반드시 필요. (예: --mark_version mark4.1)")
        sys.exit(1) # 오류 코드로 종료
        
    """
    [Deprecated: mark2.x/임시 수정본의 단일 디렉터리 처리 방식]
    아래 코드는 input_dir/output_dir를 세 번 연속 재할당하여 최종적으로 'data/val'만 처리하는 버그가 있음.
    또한 .mp3/.flac까지 포함하는데, mark4.x에서는 .wav만 사용하므로 불필요.

    input_dir = os.path.join(BASE_DIR, "data/test")  # check needed
    output_dir = os.path.join(BASE_DIR, "data/test")
    
    input_dir = os.path.join(BASE_DIR, "data/train")
    output_dir = os.path.join(BASE_DIR, "data/train")
    
    input_dir = os.path.join(BASE_DIR, "data/val")
    output_dir = os.path.join(BASE_DIR, "data/val")
    """

    # [변경] mark4.x 구조: data/{train,val,test} 각각을 순회하며 in-place(.wav만) 처리
    PROJECT_ROOT = os.path.dirname(BASE_DIR)
    data_root = os.path.join(PROJECT_ROOT, "data")
    splits = ["train", "val", "test"]

    if not os.path.isdir(data_root):
        print(f"[CRITICAL ERROR] 입력 베이스 폴더를 찾을 수 없습니다: {data_root}")
        print("폴더 구조가 '.../mark4.1/data/{train|val|test}' 형태인지 확인해주세요.")
        sys.exit(1)

    total_files = 0
    total_success = 0

    for split in splits:
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            print(f"[Warning] 분할 폴더를 찾을 수 없어 건너뜁니다: {split_dir}")
            continue

        # mark4.x는 .wav만 사용
        file_list = [f for f in os.listdir(split_dir) if f.lower().endswith(".wav")]

        if not file_list:
            print(f"[Warning] 입력 폴더에 처리할 .wav 파일이 없습니다: {split_dir}")
            continue

        print(f"[INFO] 오디오 길이 보정 시작: '{split_dir}' -> in-place")
        success_count = 0
        for fname in tqdm(file_list, desc=f"Processing {mark_version} [{split}]", unit="file"):
            in_path = os.path.join(split_dir, fname)
            out_path = in_path  # in-place
            if fix_wav_length(in_path, out_path):
                success_count += 1

        print(f"[DONE] {split} 완료: 총 {len(file_list)}개 중 {success_count}개 성공.")
        total_files += len(file_list)
        total_success += success_count

    print(f"\n[DONE] 전체 오디오 보정 완료. 총 {total_files}개 파일 중 {total_success}개 성공.")

if __name__ == "__main__":
    # 스크립트 실행 시 args.mark_version 값을 process_all 함수에 전달
    process_all(mark_version=args.mark_version)
    