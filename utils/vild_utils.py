# vild_utils.py

import torch
import torch.nn.functional as F

def normalize_mel_shape(mel_tensor, target_shape=(1, 64, 101)):
    """
    mel spectrogram을 모델 입력 형식 (C, H, W)에 맞게 변환.
    작거나 큰 입력 모두 처리하며, 필요시 zero-padding 또는 잘라냄.

    Args:
        mel_tensor (torch.Tensor): [n_mels, time], [1, n_mels, time], [1, 1, n_mels, time] 등
        target_shape (tuple): 원하는 최종 shape (기본: (1, 64, 101))

    Returns:
        torch.Tensor or None: 정규화된 mel tensor, 실패 시 None 반환
    """
    try:
        if mel_tensor is None:
            raise ValueError("mel_tensor is None")

        # 1. squeeze 가능한 차원 제거
        mel_tensor = mel_tensor.squeeze()

        # 2. 차원 확인 (2D여야 함)
        if mel_tensor.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got shape: {mel_tensor.shape}")

        n_mels, time_dim = mel_tensor.shape
        target_channels, target_mels, target_time = target_shape

        # 3. 필요한 경우 zero-padding
        pad_mel = max(0, target_mels - n_mels)
        pad_time = max(0, target_time - time_dim)

        if pad_mel > 0 or pad_time > 0:
            mel_tensor = F.pad(mel_tensor, (0, pad_time, 0, pad_mel))  # right, bottom padding

        # 4. 자르기 (너무 크면 자름)
        mel_tensor = mel_tensor[:target_mels, :target_time]  # 정확히 (64, 101)

        # 5. 차원 확장 (C, H, W)
        mel_tensor = mel_tensor.unsqueeze(0)  # (1, 64, 101)

        # 6. 최종 검증
        if mel_tensor.shape != target_shape:
            raise ValueError(f"Final shape mismatch: got {mel_tensor.shape}, expected {target_shape}")

        return mel_tensor

    except Exception as e:
        print(f"[normalize_mel_shape Error] {e}")
        return None
    