"""
shared_vild/cache_schema.py
Mark4.x / Mark5.0 공통 Feature/Metadata Cache 경로 스키마.

cache 디렉토리 구조:
  <project_root>/cache/<mark_version>/
    feature_<split>.pt    : segment-level teacher feature tensors
    metadata_<split>.pt   : segment-level metadata list (path, segment_index, saliency, ...)
"""

import os


def get_cache_root(project_root: str, mark_version: str) -> str:
    return os.path.join(project_root, "cache", mark_version)


def get_feature_cache_path(project_root: str, mark_version: str, split: str) -> str:
    return os.path.join(get_cache_root(project_root, mark_version), f"feature_{split}.pt")


def get_metadata_cache_path(project_root: str, mark_version: str, split: str) -> str:
    return os.path.join(get_cache_root(project_root, mark_version), f"metadata_{split}.pt")
