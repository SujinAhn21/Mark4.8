# convert_m4a_to_wav.py
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(BASE_DIR, 'utils')
sys.path.append(UTILS_DIR)

try:
    from archive.convert_utils import process_audio_files
    from seed_utils import set_seed
except ImportError as e:
    print(f"[IMPORT ERROR] utils 폴더 내 모듈을 불러올 수 없습니다: {e}")
    sys.exit(1)

def convert_all_audio_files(input_dir=None, output_dir=None):
    if input_dir is None:
        input_dir = os.path.join(BASE_DIR, "data")
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "data_wav")

    print(f"[INFO] 변환 시작: {input_dir} -> {output_dir}")
    try:
        process_audio_files(input_dir, output_dir)
        print(f"[DONE] 변환 완료.")
    except Exception as e:
        print(f"[ERROR] 변환 실패: {e}")

if __name__ == "__main__":
    set_seed(42)
    convert_all_audio_files()
