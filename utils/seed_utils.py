# seed_utils.py

import random
import numpy as np
import torch
import os

def set_seed(seed_value=42):
    """
    Reproducibility(재현성)를 위해 전역 random seed를 설정하는 함수.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

    """더 엄격한 재현성을 원할 경우 아래 내용 활성화가능."""
    # 성능 저하가 발생할 수 있으므로 필요 시에만 사용.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # Python의 해시 동작도 seed로 고정하려면 아래 내용 추가 가능.
    # os.environ['PYTHONHASHSEED'] = str(seed_value)

    print(f"Global seed set to {seed_value}")
    