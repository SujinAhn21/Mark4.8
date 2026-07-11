import os
import sys

import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as AF

VILD_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(VILD_DIR)
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
for p in (UTILS_DIR, VILD_DIR):
    if p not in sys.path:
        sys.path.append(p)

from parser_utils import load_audio_file


def _normalize_visual_tensor(tensor, target_shape):
    if tensor is None:
        return None

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        return None

    target_channels, target_mels, target_time = target_shape
    channels, n_mels, time_dim = tensor.shape
    if channels != target_channels:
        if channels > target_channels:
            tensor = tensor[:target_channels]
        else:
            pad_channels = torch.zeros((target_channels - channels, n_mels, time_dim), dtype=tensor.dtype)
            tensor = torch.cat([tensor, pad_channels], dim=0)

    pad_mel = max(0, target_mels - tensor.shape[1])
    pad_time = max(0, target_time - tensor.shape[2])
    if pad_mel > 0 or pad_time > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_time, 0, pad_mel))
    tensor = tensor[:, :target_mels, :target_time]

    if tensor.shape != target_shape:
        return None
    return tensor


class BaseAudioParser:
    """Shared mel-segmentation parser for teacher/student consistency."""

    def __init__(self, config):
        self.config = config
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.fft_size,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
        )
        self.amplitude_to_db = T.AmplitudeToDB()
        self.resampler_cache = {}

        # [수정 2026-07-11] torchaudio 최신버전(코랩 GPU 환경 등)에서 set_audio_backend가 아예
        # 제거되어 AttributeError가 남 — RuntimeError만 잡던 기존 코드가 이를 못 잡고 죽었음(실측).
        # 최신 torchaudio는 백엔드 자동선택이라 이 호출 자체가 불필요하므로, 있을 때만 시도한다.
        if hasattr(torchaudio, "set_audio_backend"):
            try:
                torchaudio.set_audio_backend("soundfile")
            except Exception:
                pass

    def _to_visual_mel(self, waveform):
        mel = self.mel_transform(waveform)
        mel_db = self.amplitude_to_db(mel)
        if mel_db.ndim != 3 or mel_db.shape[1] != self.config.n_mels:
            return None
        return mel_db

    def _build_visual_views(self, mel_segment):
        base = mel_segment.squeeze(0)
        if self.config.visual_view_type == "mel":
            stacked = base.unsqueeze(0)
        elif self.config.visual_view_type == "mel_delta":
            delta1 = AF.compute_deltas(base.unsqueeze(0)).squeeze(0)
            delta2 = AF.compute_deltas(delta1.unsqueeze(0)).squeeze(0)
            stacked = torch.stack([base, delta1, delta2], dim=0)
        elif self.config.visual_view_type == "mel_energy":
            energy = (base - base.mean()) / (base.std() + 1e-6)
            stacked = torch.stack([base, energy], dim=0)
        else:
            stacked = base.unsqueeze(0)
        return stacked

    def _compute_saliency_score(self, mel_segment):
        base = mel_segment.squeeze(0)
        energy = base.abs().mean()
        if base.shape[-1] > 1:
            flux = (base[:, 1:] - base[:, :-1]).abs().mean()
        else:
            flux = torch.tensor(0.0, dtype=base.dtype)
        return float(energy + 0.5 * flux)

    def load_and_segment(self, file_path):
        waveform = load_audio_file(file_path, self.config.sample_rate, self.resampler_cache)
        if waveform is None or waveform.numel() == 0:
            print(f"[Parser] Skipped unreadable file: {file_path}")
            return []

        try:
            mel_db = self._to_visual_mel(waveform)
            if mel_db is None:
                print(f"[Parser] Unexpected mel shape from {file_path}")
                return []

            _, _, total_time = mel_db.shape
            window = self.config.segment_length
            stride = self.config.segment_hop
            max_segments = getattr(self.config, "max_segments", 5)

            if total_time < window:
                print(f"[Parser] Mel too short for segmentation: {total_time} < {window} in {file_path}")
                return []

            candidates = []
            for start in range(0, total_time - window + 1, stride):
                segment = mel_db[:, :, start:start + window]
                views = self._build_visual_views(segment)
                normed = _normalize_visual_tensor(
                    views,
                    (self.config.num_input_channels, self.config.n_mels, self.config.segment_length),
                )
                if normed is not None:
                    candidates.append((start, self._compute_saliency_score(segment), normed))

            if not candidates:
                print(f"[Parser] No valid segments from: {file_path}")
                return []

            if self.config.segment_selection_mode == "salient_topk":
                selected = sorted(candidates, key=lambda x: x[1], reverse=True)[:max_segments]
                selected.sort(key=lambda x: x[0])
            else:
                selected = candidates[:max_segments]

            segments = [item[2] for item in selected]
            if len(segments) < max_segments:
                last_valid = segments[-1]
                segments += [last_valid.clone() for _ in range(max_segments - len(segments))]

            return segments
        except Exception as e:
            print(f"[Parser] Exception while parsing {file_path}: {e}")
            return []

    def parse_sample(self, file_path, label_text):
        segments = self.load_and_segment(file_path)
        segments = [seg for seg in segments if isinstance(seg, torch.Tensor)]
        if not segments:
            raise ValueError(f"[Parser] No mel segments from {file_path}")

        mel_tensor = torch.cat(segments, dim=0)
        label_idx = self.config.get_class_index(label_text)
        return mel_tensor, label_idx
