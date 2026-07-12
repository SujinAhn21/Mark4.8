# audio_utils.py

import torch
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def prepare_teacher_embedding(embedding: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Teacher 임베딩 텐서를 [1, D] 형태로 표준화합니다.

    Args:
        embedding (torch.Tensor): 다양한 형태의 입력 텐서 (예: [1,1,1,D], [1,D], [D])
        device (torch.device): 텐서를 이동시킬 디바이스

    Returns:
        torch.Tensor: [1, D] 형식의 임베딩 텐서

    Raises:
        ValueError: 예상하지 못한 텐서 shape일 경우
    """
    embedding = embedding.to(device)
    original_shape = embedding.shape

    if embedding.dim() > 2:
        embedding = embedding.view(-1)  # 다차원 -> 1D 평탄화
        logging.info(f"Flattened teacher embedding from {original_shape} to {embedding.shape}")
    elif embedding.dim() == 2 and embedding.shape[0] == 1:
        embedding = embedding.squeeze(0)
    elif embedding.dim() == 1:
        pass  # 이미 1D
    else:
        raise ValueError(f"Unexpected teacher embedding shape: {original_shape}")

    final_embedding = embedding.reshape(1, -1)  # 최종 [1, D] 형식 보장
    logging.info(f"Final teacher embedding shape: {final_embedding.shape}")
    return final_embedding
    