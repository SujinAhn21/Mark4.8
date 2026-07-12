# train.py

"""
[Deprecated: utils 경로(model/utils) 가정 및 student 함수명 불일치]
import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(BASE_DIR, "utils")
sys.path.append(UTILS_DIR)

from seed_utils import set_seed
from teacher_train import train_teacher
from student_train_distillation import train_student
"""
# ========================= 변경 적용 =========================

import argparse
import os
import sys

# 경로 정합성
BASE_DIR = os.path.dirname(os.path.abspath(__file__))            # model/
PROJECT_ROOT = os.path.dirname(BASE_DIR)                         # mark4.1/
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
for p in (PROJECT_ROOT, UTILS_DIR):
    if p not in sys.path:
        sys.path.append(p)

from seed_utils import set_seed
from teacher_train import train_teacher
# 호환: student_train_distillation.py 내부에 train_student 래퍼 정의
from student_train_distillation import train_student

def main():
    parser = argparse.ArgumentParser(description="Train Teacher or Student model.")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['teacher', 'student'],
        required=True,
        help="학습 모드 선택 (teacher 또는 student)"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="전역 랜덤 시드 값"
    )
    parser.add_argument(
        '--mark_version',
        type=str,
        default="mark4.1",
        help="모델 및 데이터셋 버전 (예: mark4.1)"
    )
    args = parser.parse_args()

    set_seed(args.seed)

    if args.mode == "teacher":
        print(f"[INFO] ViLD-text Teacher 모델 ({args.mark_version}) 학습을 시작합니다. (Seed: {args.seed})")
        train_teacher(seed_value=args.seed, mark_version=args.mark_version)

    elif args.mode == "student":
        print(f"[INFO] ViLD-image Student 모델 ({args.mark_version}) 학습을 시작합니다. (Seed: {args.seed})")
        train_student(seed_value=args.seed, mark_version=args.mark_version)

if __name__ == "__main__":
    main()
    