# vild_losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from vild_config import AudioViLDConfig

class ViLDLosses:
    """
    ViLD 기반 모델 학습을 위한 커스텀 손실 함수 모듈

    주요 기능:
    - ViLD-text: 텍스트 임베딩 기반 CrossEntropyLoss
    - ViLD-image: 오디오 임베딩 간 거리 기반 L1 + Cosine Distance MSE 혼합 손실
    """

    def __init__(self, config: AudioViLDConfig):
        self.text_loss_weight = config.text_loss_weight
        self.image_loss_weight = config.image_loss_weight
        self.ce_loss = nn.CrossEntropyLoss()

    def compute_text_loss(self, logits, targets):
        """
        텍스트 기반 분류 손실 (CrossEntropyLoss)
        
        Args:
            logits (Tensor): [B, C+1], cosine similarity 기반 예측 결과
            targets (Tensor): [B], 정답 클래스 인덱스

        Returns:
            Tensor: weighted text loss
        """
        return self.text_loss_weight * self.ce_loss(logits, targets)

    def compute_image_loss(self, student_proj, teacher_embeddings):
        """
        이미지(오디오) 임베딩 간 유사도 손실
        
        Args:
            student_proj (Tensor): [B, D], student 모델 임베딩
            teacher_embeddings (Tensor): [B, D], teacher soft label 임베딩

        Returns:
            Tensor: weighted hybrid distance loss
        """
        l1 = F.l1_loss(student_proj, teacher_embeddings)
        cos_sim = F.cosine_similarity(student_proj, teacher_embeddings, dim=1)
        cos_dist = 1 - cos_sim
        cos_mse = torch.mean(cos_dist ** 2)
        return self.image_loss_weight * (0.5 * l1 + 0.5 * cos_mse)

    def total_loss(self, logits, targets, student_proj, teacher_embeddings):
        """
        전체 손실 계산 (ViLD-text + ViLD-image)

        Returns:
            Tuple[Tensor, Tensor, Tensor]: (total, text, image) loss
        """
        text_loss = self.compute_text_loss(logits, targets)
        image_loss = self.compute_image_loss(student_proj, teacher_embeddings)
        total = text_loss + image_loss
        return total, text_loss, image_loss
    