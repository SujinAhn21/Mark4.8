# model/profile_efficiency.py
"""
Mark 모델(specialist)의 효율성 지표(파라미터 수, 모델 용량, FLOPs, CPU 추론 지연시간)를 측정한다.
비교모델(PANNs/AST/BEATs/EfficientAT/ConvNeXt-audio/EAT/CED-Tiny)과의
효율성-정확도 산점도에 쓸 숫자를 낸다.

주의:
- 텍스트 임베딩(SentenceTransformer)은 클래스에 대해 배포 시 1회만 계산해
  캐싱하는 게 정상적인 엣지 배포 방식이므로, 이 스크립트는 그 비용을 추론 지연시간에서 제외한다
  (오디오 1개가 들어올 때마다 반복되는 비용이 아니기 때문).
- FLOPs는 encoder+branch_head(둘 다 nn.Linear/Conv2d 등 thop이 인식하는 레이어로 구성)만 측정한다.
  text_head(ViLDTextHead)는 nn.Linear 없이 raw matmul(cosine 유사도)만 쓰는데, 연산량이
  384×2 수준으로 미미해 무시 가능.
"""

import argparse
import json
import os
import sys
import time

import torch
from thop import profile as thop_profile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
VILD_DIR = os.path.join(PROJECT_ROOT, "vild")
for p in (PROJECT_ROOT, UTILS_DIR, VILD_DIR):
    if p not in sys.path:
        sys.path.append(p)

from vild_config import AudioViLDConfig
from vild_model import LearnableBackgroundEmbedding, SimpleAudioEncoder, ViLDTextHead
from vild_head import DualBranchStudentHead


def count_parameters(*modules):
    total = 0
    for m in modules:
        if m is None:
            continue
        total += sum(p.numel() for p in m.parameters())
    return total


class _EncoderBranchWrapper(torch.nn.Module):
    """encoder+branch_head를 하나로 묶어 thop이 한 번의 forward로 FLOPs를 잴 수 있게 함."""

    def __init__(self, encoder, branch_head):
        super().__init__()
        self.encoder = encoder
        self.branch_head = branch_head

    def forward(self, x):
        feat = self.encoder(x)
        return self.branch_head(feat)


def profile(mark_version: str, num_warmup: int = 10, num_runs: int = 100):
    device = torch.device("cpu")  # 엣지 배포 타겟이 CPU이므로 지연시간은 항상 CPU 기준으로 측정
    config = AudioViLDConfig(mark_version=mark_version)

    encoder = SimpleAudioEncoder(config).to(device).eval()
    branch_head = DualBranchStudentHead(config.embedding_dim).to(device).eval()
    text_head = ViLDTextHead(config).to(device).eval()
    background_embedding = None
    if config.use_background_embedding:
        background_embedding = LearnableBackgroundEmbedding(config.embedding_dim).to(device).eval()

    total_params = count_parameters(encoder, branch_head, text_head, background_embedding)
    model_size_mb = total_params * 4 / (1024 ** 2)  # fp32 가정

    num_classes = len(config.get_classes_for_text_prompts())
    dummy_segment = torch.randn(1, config.num_input_channels, config.n_mels, config.segment_length, device=device)
    dummy_text_emb = torch.randn(num_classes, config.embedding_dim, device=device)

    # [추가] FLOPs 측정 (encoder+branch_head, 세그먼트 1개 기준).
    # thop.profile은 프로파일링 대상 모듈에 total_ops/total_params 훅을 남겨 이후 재사용 시
    # 충돌(AttributeError)을 일으키므로, 지연시간 측정용 encoder/branch_head와 별개의
    # 인스턴스를 새로 만들어 여기서만 쓰고 버린다.
    flops_encoder = SimpleAudioEncoder(config).to(device).eval()
    flops_branch_head = DualBranchStudentHead(config.embedding_dim).to(device).eval()
    wrapper = _EncoderBranchWrapper(flops_encoder, flops_branch_head)
    macs, _ = thop_profile(wrapper, inputs=(dummy_segment,), verbose=False)
    gmacs_per_segment = macs / 1e9
    gflops_per_segment = gmacs_per_segment * 2  # 통상 FLOPs ≈ 2 x MACs 관례

    def run_once():
        with torch.no_grad():
            base_features = encoder(dummy_segment)
            supervised_features, distill_features = branch_head(base_features)
            _ = text_head(supervised_features, dummy_text_emb)
            _ = text_head(distill_features, dummy_text_emb)

    for _ in range(num_warmup):
        run_once()

    start = time.perf_counter()
    for _ in range(num_runs):
        run_once()
    elapsed = time.perf_counter() - start

    ms_per_segment = (elapsed / num_runs) * 1000
    ms_per_clip = ms_per_segment * config.max_segments  # 5초 클립 = max_segments개 세그먼트

    result = {
        "mark_version": mark_version,
        "total_parameters": total_params,
        "model_size_mb": round(model_size_mb, 4),
        "gmacs_per_segment": round(gmacs_per_segment, 6),
        "gflops_per_segment_approx": round(gflops_per_segment, 6),
        "gflops_per_5s_clip_approx": round(gflops_per_segment * config.max_segments, 6),
        "cpu_latency_ms_per_segment": round(ms_per_segment, 4),
        "cpu_latency_ms_per_5s_clip": round(ms_per_clip, 4),
        "num_warmup": num_warmup,
        "num_runs": num_runs,
        "note": (
            "텍스트 임베딩(SentenceTransformer) 계산 비용은 배포 시 1회 캐싱 가정으로 제외됨. "
            "FLOPs는 encoder+branch_head만 측정(text_head는 미미해 제외), FLOPs≈2×MACs 근사."
        ),
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    plot_dir = os.path.join(PROJECT_ROOT, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    out_path = os.path.join(plot_dir, f"efficiency_profile_{mark_version}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 효율성 프로파일 저장: {out_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mark specialist 모델의 파라미터 수·모델 용량·FLOPs·CPU 추론 지연시간을 측정합니다.")
    parser.add_argument("--mark_version", type=str, required=True)
    parser.add_argument("--num_warmup", type=int, default=10)
    parser.add_argument("--num_runs", type=int, default=100)
    args = parser.parse_args()
    profile(mark_version=args.mark_version, num_warmup=args.num_warmup, num_runs=args.num_runs)
