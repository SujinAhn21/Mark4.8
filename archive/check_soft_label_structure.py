# check_soft_label_structure.py
import os
import pickle
import argparse

def check_soft_label_structure(mark_version="mark4.1"):  # change needed
    pkl_path = f"soft_labels_{mark_version}.pkl"

    if not os.path.exists(pkl_path):
        print(f"[ERROR] 파일이 존재하지 않습니다: {pkl_path}")
        return

    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"[ERROR] 파일 로딩 중 오류 발생: {e}")
        return

    print(f"\n[INFO]'{pkl_path}' 불러오기 성공.")
    print(f"[INFO] 총 샘플 개수: {len(data)}")

    if len(data) == 0:
        print("[WARNING] 데이터가 비어 있습니다.")
        return

    print("\n[INFO] 첫 번째 항목 구조 예시:")
    first = data[0]
    if isinstance(first, dict):
        for k, v in first.items():
            if hasattr(v, 'shape'):
                print(f"  - {k}: tensor {tuple(v.shape)}")
            else:
                print(f"  - {k}: {type(v)} | {str(v)[:60]}...")
    else:
        print(first)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mark_version", type=str, default="mark4.1")  # change needed
    args = parser.parse_args()

    check_soft_label_structure(args.mark_version)
