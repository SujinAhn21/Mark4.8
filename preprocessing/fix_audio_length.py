# fix_audio_length.py 
# 인자 반영

import os
import sys
import torch
import torchaudio
import argparse
from tqdm import tqdm

# === 인자 parsing: default 값을 None으로 변경하여 인자 전달 여부 확인 ===
# [변경 2026-07-11] "N초로 강제 통일(자르기+패딩)" -> "최소 길이만 보장(패딩만, 자르지 않음)"으로 재설계.
# 이유: 파서(vild_parser_common.py)가 세그먼트 단위(1초 창, 0.5초 stride)로 salient_topk 5개를 뽑도록
# 이미 자체 정규화하고 있어서, 원본 클립 길이를 사전에 통일할 구조적 이유가 없음. 오히려 예전 방식(15초로
# 자르기)은 15초 넘는 클립의 뒷부분을 통째로 버려 타겟 소리가 잘려나가는 실제 버그였음(FSD50K 실측:
# 15초 초과 클립 17.6%). 반대로 너무 짧으면 파서가 파일을 통째로 버리거나(1.01초 미만) 세그먼트를
# 5개 못 채워 마지막 세그먼트를 복제하는 열화 폴백이 걸림 -> 최소 길이 보장만 하고 자르기는 없앰.
parser = argparse.ArgumentParser(
    description="오디오 파일이 최소 3초(48000 샘플) 미만이면 무음 패딩으로 채웁니다(자르기 없음)."
)  # 전처리.
parser.add_argument("--mark_version", type=str, default=None, 
                    help="모델 버전 (예: mark4.1). 이 버전에 따라 입/출력 폴더가 결정됩니다.")
args = parser.parse_args()

# === 기본 경로 설정 ====
# 현재 스크립트가 있는 폴더가 기준이 됨 ( /content/MyProject/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === 보정 파라미터 ===
TARGET_SAMPLE_RATE = 16000 # sample rate 도 제한을 해서 주파수 해상도를 일정하게 맞추기
# [변경 2026-07-11] TARGET_NUM_SAMPLES(강제 통일 길이) -> TARGET_MIN_SAMPLES(최소 보장 길이)로 의미 변경.
# 근거: vild_config.py 기준 segment_length=101프레임, segment_hop=50프레임, max_segments=5.
# salient_topk가 "서로 다른" 5개 세그먼트를 뽑으려면 최소 101 + 4*50 = 301프레임이 필요하고,
# hop_length=160/sample_rate=16000이므로 301프레임 = 300*160 = 48,000샘플(정확히 3.0초).
# 이보다 짧으면 세그먼트가 5개 미만이라 파서가 마지막 세그먼트를 복제해 채우는 폴백이 걸림(열화).
TARGET_MIN_SAMPLES = 16000 * 3  # 48,000 samples (3.0초 = 5개 서로 다른 세그먼트를 보장하는 최소 길이)

def fix_wav_length(wav_path, save_path):
    try:
        waveform, sr = torchaudio.load(wav_path)

        # 샘플레이트 맞추기(보정하기)
        if sr != TARGET_SAMPLE_RATE:
            resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)
            waveform = resample(waveform)

        # [변경] 짧으면 최소 길이까지만 무음 패딩. 길어도 자르지 않음(뒷부분 소리 소실 방지).
        num_samples = waveform.shape[1]
        if num_samples < TARGET_MIN_SAMPLES:
            pad_len = TARGET_MIN_SAMPLES - num_samples
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
    