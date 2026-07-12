"""
shared_vild/checkpoint_utils.py
Mark4.x / Mark5.0 공통 Checkpoint 저장/로드 유틸리티.

저장 스키마:
  schema_version    : int
  model_type        : str  ("teacher_encoder", "teacher_classifier", "student_full", ...)
  mark_version      : str  ("mark4.1", "mark5.0", ...)
  config_metadata   : dict (encoder_type, visual_view_type, num_input_channels, segment 설정 등)
  model_state_dict  : OrderedDict
  branch_state_dict : OrderedDict | None
  classifier_state_dict : OrderedDict | None
  text_embeddings   : Tensor | None
  background_embedding_state_dict : OrderedDict | None
"""

import os
import torch

SCHEMA_VERSION = 1

_CONFIG_KEYS = [
    "encoder_type",
    "visual_view_type",
    "num_input_channels",
    "embedding_dim",
    "segment_length",
    "segment_hop",
    "max_segments",
    "n_mels",
    "sample_rate",
    "fft_size",
    "hop_length",
]


def extract_config_metadata(config) -> dict:
    meta = {}
    for key in _CONFIG_KEYS:
        val = getattr(config, key, None)
        if val is not None:
            meta[key] = val
    return meta


def apply_config_metadata(config, ckpt: dict):
    meta = ckpt.get("config_metadata", {})
    for key, val in meta.items():
        if hasattr(config, key):
            setattr(config, key, val)


def save_checkpoint(
    path: str,
    model_type: str,
    mark_version: str,
    model_state=None,
    branch_state=None,
    classifier_state=None,
    text_embeddings=None,
    config=None,
    background_embedding_state=None,
):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_type": model_type,
        "mark_version": mark_version,
        "config_metadata": extract_config_metadata(config) if config is not None else {},
        "model_state_dict": model_state,
        "branch_state_dict": branch_state,
        "classifier_state_dict": classifier_state,
        "text_embeddings": text_embeddings,
        "background_embedding_state_dict": background_embedding_state,
    }
    torch.save(payload, path)


def load_checkpoint(path: str, map_location=None) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[checkpoint_utils] Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=map_location)
    if not isinstance(ckpt, dict):
        return {"model_state_dict": ckpt}
    return ckpt


def resolve_state_dict(ckpt: dict, *keys):
    for key in keys:
        if key in ckpt and ckpt[key] is not None:
            return ckpt[key]
    raise KeyError(
        f"[checkpoint_utils] None of the keys {keys} found in checkpoint. "
        f"Available keys: {list(ckpt.keys())}"
    )
