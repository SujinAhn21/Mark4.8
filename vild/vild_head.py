# vild_head.py

import torch
import torch.nn as nn

class ViLDHead(nn.Module):
    """
    Student 모델의 region embedding을 텍스트 임베딩 공간과 동일한 차원으로 투영(projection)하는 헤드

    - CrossEntropyLoss 기반 분류용으로 cosine 정규화는 제거함
    - 구조: Linear -> LayerNorm -> ReLU
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.projection(x)
        x = self.norm(x)
        x = self.activation(x)
        return x  # 정규화 제거됨


class DualBranchStudentHead(nn.Module):
    """Separates supervised and distillation projections for student training."""

    def __init__(self, embedding_dim):
        super().__init__()
        self.supervised_branch = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.distill_branch = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, features):
        supervised_features = self.supervised_branch(features)
        distill_features = self.distill_branch(features)
        return supervised_features, distill_features
    
